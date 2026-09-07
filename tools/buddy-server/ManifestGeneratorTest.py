#!/usr/bin/env python3
# ===- ManifestGeneratorTest.py - Manifest generator tests -------------===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===----------------------------------------------------------------------===//


import importlib.util
import json
import pathlib
import subprocess
import sys


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: test.py WHISPER_GENERATOR GENERIC_GENERATOR SPEC RAX_PACK OUT_DIR"
        )

    whisper_path, generic_path, spec_path, rax_pack, output_path = map(
        pathlib.Path, sys.argv[1:]
    )
    output_path.mkdir(parents=True, exist_ok=True)

    whisper_generator = load_module(whisper_path, "whisper_manifest_generator")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    whisper_text = whisper_generator.gen_manifest(
        spec, "file:whisper_runner.so", "file:whisper_transcription.so"
    )
    assert 'runner_library = "file:whisper_runner.so"' in whisper_text
    assert (
        'transcription_library = "file:whisper_transcription.so"'
        in whisper_text
    )
    for key in (
        "params_size",
        "vocab_size",
        "max_token_len",
        "mel_bins",
        "audio_frames",
        "enc_seq",
        "enc_dim",
        "sot_token",
        "eot_token",
    ):
        assert f'{key} = "{spec[key]}"' in whisper_text

    manifest = output_path / "whisper.mlir"
    manifest.write_text(whisper_text, encoding="utf-8")
    for filename in (
        spec.get("weight_file", "arg0.data"),
        spec.get("so_name", "whisper_model.so"),
        spec.get("vocab_file", "vocab.txt"),
        "whisper_runner.so",
        "whisper_transcription.so",
    ):
        (output_path / filename).write_bytes(b"test")
    subprocess.run(
        [str(rax_pack), str(manifest), "-o", str(output_path / "whisper.rax")],
        check=True,
        capture_output=True,
        text=True,
    )

    generic_generator = load_module(generic_path, "generic_manifest_generator")
    generic_config = {
        "model_id": "test",
        "model_family": "test",
        "shape": {
            "head_num": 1,
            "max_token_len": 2,
            "hidden_size": 1,
            "vocab_size": 2,
            "kv_layers": 0,
        },
        "tokens": {"vocab_file": "vocab.txt"},
        "weights": [
            {
                "tag": "params",
                "mlir_type": "f32",
                "num_elements": 1,
                "file": "w.bin",
            }
        ],
        "compilation": {"so_name": "model.so"},
        "mlir_types": {"kv": "f32", "logits": "f32"},
    }
    without_plugin = generic_generator.gen_manifest(generic_config)
    assert "transcription_library" not in without_plugin
    with_plugin = generic_generator.gen_manifest(
        generic_config, transcription_library="transcription.so"
    )
    assert 'transcription_library = "file:transcription.so"' in with_plugin

    print("Manifest generator tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
