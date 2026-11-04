# RUN: %PYTHON %s

"""Exercise the MMS build entry point and the packaged vocoder ABI."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(os.environ["BUDDY_SRC_ROOT"])
spec_path = root / "models/mms/specs/f32.json"
spec = json.loads(spec_path.read_text())

with tempfile.TemporaryDirectory() as directory:
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools/buddy-codegen/build_model.py"),
            "--spec",
            str(spec_path),
            "--build-dir",
            "build",
            "--dry-run",
        ],
        cwd=directory,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "-DBUDDY_BUILD_MMS_MODEL=ON" in result.stderr
    assert f"-DBUDDY_MMS_SPEC={spec_path}" in result.stderr
    assert f"--build {directory}/build --target mms_rax" in result.stderr

module_spec = importlib.util.spec_from_file_location(
    "gen_mms_manifest", root / "models/mms/codegen/gen_mms_manifest.py"
)
generator = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(generator)
manifest = generator.gen_manifest(spec, "mms_runner.so")
for attribute, value in (
    ("max_seq_len", spec["max_seq_len"]),
    ("flow_size", spec["flow_size"]),
    ("params_size", spec["params_size"]),
    ("audio_rate", 256),
    ("audio_buffer_size", spec["max_seq_len"] * 256),
):
    assert f'{attribute} = "{value}"' in manifest, attribute
assert "tensor<1x192x30xf32>" in manifest
assert "tensor<1x7680xf32>" in manifest
assert 'runner_library = "file:mms_runner.so"' in manifest
with_dependencies = generator.gen_manifest(
    spec,
    "mms_runner.so",
    ["libomp.so", "file:libmlir_c_runner_utils.so"],
    version="1.2.3",
)
assert 'version = "1.2.3"' in with_dependencies
assert 'uri = "file:libomp.so"' in with_dependencies
assert 'uri = "file:libmlir_c_runner_utils.so"' in with_dependencies
assert "@runtime_dep_1 {id = 2 : i32" in with_dependencies
assert "@runtime_dep_2 {id = 3 : i32" in with_dependencies
manifest_cli = subprocess.run(
    [
        sys.executable,
        str(root / "models/mms/codegen/gen_mms_manifest.py"),
        "--spec",
        str(spec_path),
        "--version",
        "1.2.3",
        "--dep-shared-lib",
        "libomp.so",
        "--dep-shared-lib",
        "file:libmlir_c_runner_utils.so",
    ],
    text=True,
    capture_output=True,
    check=True,
)
assert manifest_cli.stdout == with_dependencies

# A second input length must change both the runtime metadata and the ABI.
short_spec = dict(spec, max_seq_len=7)
short_manifest = generator.gen_manifest(short_spec, "mms_runner.so")
assert 'max_seq_len = "7"' in short_manifest
assert 'audio_buffer_size = "1792"' in short_manifest
assert "tensor<1x192x7xf32>" in short_manifest
assert "tensor<1x1792xf32>" in short_manifest

for changes in (
    {"max_seq_len": 0},
    {"flow_size": -1},
    {"params_size": 0},
    {"upsample_rates": []},
    {"upsample_rates": [8, 8, 0, 2]},
):
    try:
        generator.gen_manifest(dict(spec, **changes), "mms_runner.so")
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted invalid MMS dimensions: {changes}")

print("MMS build entry point, manifest ABI and invalid dimensions: PASS")
