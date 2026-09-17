#!/usr/bin/env python3
"""Relink the validated 1-layer/16+8 model with explicit kernel replacements.

This does not re-import the graph, regenerate its adapters/reference, rebuild
Triton kernels, or operate the FPGA. Baseline kernels come from the stage's
immutable archive, not a possibly changed shared Triton build directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


MODEL = Path(__file__).resolve().parents[1]
REPO = MODEL.parents[3]
TOOLS = MODEL / "tools"
PROMPT = "What is France?"
PROMPT_IDS = [151644, 872, 198, 3838, 374, 9625, 30, 151645,
              198, 151644, 77091, 198, 151667, 271, 151668, 271]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def baseline_files(stage):
    """Validate the exact archived baseline and describe a regular build tree."""
    archive_dir = stage / "model-lib"
    manifest = read(archive_dir / "archive.json")
    require(manifest.get("case_count") == 46 and len(manifest["cases"]) == 46,
            "expected the validated 46-kernel 1-layer baseline archive")
    require(manifest.get("contains_runtime") is False and
            manifest.get("contains_test_launch") is False,
            "baseline archive contains runtime or launch objects")
    require(digest(archive_dir / "libqwen_triton.a") == manifest["sha256"],
            "baseline archive hash mismatch")
    files, names = [], []
    for case in manifest["cases"]:
        name = case["case"]
        require(re.fullmatch(r"[a-z][a-z0-9_]*", name) and name not in names,
                "invalid or duplicate baseline case name")
        names.append(name)
        evidence = archive_dir / "evidence" / name
        for filename, expected in case["files_sha256"].items():
            require(Path(filename).name == filename, "invalid archived filename")
            source = (archive_dir / (name + "." + filename)
                      if filename in ("kernel.o", "adapter.o") else evidence / filename)
            require(source.is_file() and digest(source) == expected,
                    "baseline evidence hash mismatch: " + str(source))
            native = filename in ("kernel.o", "adapter.o", "kernel.ll",
                                  "kernel.s", "kernel.nr.S")
            relative = Path(name) / ("nr" if native else "") / filename
            files.append((source, relative, expected))
        source = evidence / "manifest.json"
        native_manifest = read(source)
        require(native_manifest.get("case", {}).get("name") == name,
                "baseline native manifest case mismatch")
        require(native_manifest.get("elf_audit", {}).get("status") == "PASS",
                "baseline kernel lacks passing ELF audit")
        files.append((source, Path(name) / "nr/manifest.json", digest(source)))
    return manifest, sorted(names), files


def replacement_files(path, baseline_case):
    """Validate replacements before dry-run/build and track their immutable bytes."""
    frontend_path = path / "frontend.json"
    native_path = path / "nr/manifest.json"
    frontend, native = read(frontend_path), read(native_path)
    before = baseline_case["frontend"]
    for field in ("name", "family", "symbol", "arguments", "signature"):
        require(frontend.get(field) == before.get(field),
                f"replacement changes external ABI/semantics: {path.name}: {field}")
        require(native.get("case", {}).get(field) == frontend.get(field),
                f"replacement native/frontend mismatch: {path.name}: {field}")
    dimensions = lambda manifest: {key: value for key, value in manifest["constexprs"].items()
                                    if key not in ("BLOCK", "BM", "BN", "BK")}
    require(dimensions(frontend) == dimensions(before),
            "replacement changes dimensions/arithmetic constants: " + path.name)
    for field in ("kernel", "constexprs", "grid"):
        require(native["case"].get(field) == frontend.get(field),
                f"replacement native/frontend mismatch: {path.name}: {field}")
    require(native.get("elf_audit", {}).get("status") == "PASS" and
            native.get("heap_allocations") == 0,
            "replacement lacks a passing native ELF audit or allocates heap: " + path.name)
    inputs = [native_path]
    for key, relative in (("frontend_manifest_sha256", "frontend.json"),
                          ("ttir_sha256", "kernel.ttir"),
                          ("linalg_sha256", "kernel.linalg.mlir"),
                          ("llvm_sha256", "nr/kernel.ll"),
                          ("adapter_sha256", "adapter.c")):
        source = path / relative
        require(source.is_file() and digest(source) == native.get(key),
                "stale replacement build: " + str(source))
        inputs.append(source)
    for relative in ("nr/kernel.o", "nr/adapter.o", "nr/kernel.s", "nr/kernel.nr.S"):
        source = path / relative
        require(source.is_file(), "replacement artifact missing: " + str(source))
        inputs.append(source)
    return inputs


def prepare(args):
    stage, output = args.stage.resolve(), args.output.absolute()
    require(not output.exists() and not output.is_symlink(),
            "output already exists; choose a fresh immutable variant directory")
    require(not output.resolve().is_relative_to(stage),
            "output must be outside the immutable baseline stage")
    graph_report = stage / "replacement/triton-call-replacement.json"
    report = read(graph_report)
    for kind in ("prefill", "decode"):
        abi = report["graphs"][kind]["entry_abi"]
        require(abi.get("result_descriptor_count") == 4 and len(abi.get("outputs", [])) == 4,
                "this helper only accepts the validated 1-layer graph ABI")
    reference = stage / "quant-nr-trace/quant-reference.json"
    metadata = read(reference)
    require(metadata.get("layers") == 1 and metadata.get("capacity") == 512 and
            metadata.get("arithmetic_profile") == "nr-fpga",
            "reference must describe one layer, context512, NR arithmetic")
    require(metadata.get("prompt_ids") == PROMPT_IDS and
            len(metadata.get("decode_steps_recorded", [])) >= 8,
            "reference does not match the fixed What is France? 16+8 trajectory")
    baseline, cases, files = baseline_files(stage)
    replacements = [path.resolve() for path in args.replace_case]
    require(len({path.name for path in replacements}) == len(replacements),
            "duplicate --replace-case")
    baseline_cases = {case["case"]: case for case in baseline["cases"]}
    replacement_inputs = []
    for path in replacements:
        require(path.name in cases and (path / "frontend.json").is_file() and
                (path / "nr/manifest.json").is_file(),
                "replacement must be one built case from the baseline: " + str(path))
        replacement_inputs += replacement_files(path, baseline_cases[path.name])

    weights = args.weights.resolve()
    segment = args.weight_manifest.resolve()
    tokenizer = args.tokenizer.resolve()
    layout = args.intermediate_layout.resolve()
    arrays = stage / "quant-nr-trace/arrays.npz"
    prefill = stage / "nr-prefill/forward_prefill.ll"
    decode = stage / "nr-decode/forward_decode.ll"
    adapters = stage / "replacement/qwen_triton_adapters.c"
    typed_graphs = [stage / "replacement" / f"subgraph0_{kind}.triton.mlir"
                    for kind in ("prefill", "decode")]
    inputs = [weights, segment, tokenizer, layout, arrays, reference,
              prefill, decode, adapters, graph_report,
              stage / "model-lib/archive.json", *typed_graphs, *replacement_inputs]
    require(all(path.is_file() for path in inputs), "one or more required input files are missing")
    referenced_cases = set(re.findall(r"\b_mlir_ciface_kernel_([a-z][a-z0-9_]*)\b", adapters.read_text()))
    require(referenced_cases == set(cases),
            "graph adapters and baseline archive do not reference the same kernel set")
    weight_record = read(segment)
    require(weights.stat().st_size == weight_record["bytes"] and
            digest(weights) == weight_record["sha256"], "weight segment hash/size mismatch")

    sys.path.insert(0, str(TOOLS))
    from build_nr_w8a8_image import resolve_linker
    linker = resolve_linker(args.linker, REPO / "llvm/build-2d26/bin")
    llvm = REPO / "llvm/build-2d26/bin"
    for tool in ("llvm-ar", "llvm-nm", "llvm-objcopy"):
        require((llvm / tool).is_file(), "missing tool: " + str(llvm / tool))
    python = [sys.executable, "-B"]
    commands = [
        [*python, TOOLS / "kernel_build_overlay.py", "--source", output / "baseline-kernels",
         *[part for path in replacements for part in ("--replace-case", path)],
         "--output", output / "kernel-overlay"],
        [*python, TOOLS / "archive_kernels.py", "--triton-build", output / "kernel-overlay",
         *[part for name in cases for part in ("--case", name)],
         "--output", output / "model-lib", "--ar", llvm / "llvm-ar", "--nm", llvm / "llvm-nm"],
        [*python, TOOLS / "build_nr_w8a8_image.py", "--repo-root", REPO,
         "--linker", linker["path"], "--report", graph_report, "--segment", segment,
         "--graph-ir", prefill, "--decode-ir", decode,
         "--archive", output / "model-lib/libqwen_triton.a", "--adapters", adapters,
         "--output", output / "image", "--layers", "1", "--cache-len", "512",
         "--prefill-len", "16", "--decode-steps", "8", "--prompt-text", PROMPT,
         "--prompt-ids", ",".join(map(str, PROMPT_IDS)), "--tokenizer-blob", tokenizer,
         "--reference-arrays", arrays, "--reference-metadata", reference,
         "--intermediate-arrays", arrays, "--intermediate-layout", layout,
         "--intermediate-graph-dir", stage / "replacement", "--intermediate-progress"],
        [*python, TOOLS / "prepare_model_run.py", "--image", output / "image/qwen_model.bin",
         "--elf", output / "image/qwen_model.elf", "--weights", weights,
         "--weight-manifest", segment, "--tokenizer", tokenizer,
         "--output", output / "prepared", "--nm", llvm / "llvm-nm", "--objcopy", llvm / "llvm-objcopy"],
    ]
    plan = {"status": "PLANNED_NOT_BUILT", "scope": "1-layer fixed16+8 only",
            "baseline_stage": str(stage), "baseline_archive_sha256": baseline["sha256"],
            "cases": cases, "replace_cases": [str(path) for path in replacements],
            "prompt_text": PROMPT, "prompt_ids": PROMPT_IDS, "profile_kernels": False,
            "intermediate_progress": True, "linker": linker,
            "replacement_contracts_and_hashes_checked": True,
            "commands": [[str(part) for part in command] for command in commands]}
    return stage, output, files, inputs, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True,
                        help="immutable validated review-native-1l stage")
    parser.add_argument("--replace-case", type=Path, action="append", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weight-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--intermediate-layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--linker", type=Path,
                        help="LLD >=20; default uses the shared image builder resolver")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate existing inputs and print commands without writing/building")
    args = parser.parse_args()
    stage, output, files, inputs, plan = prepare(args)
    for command in plan["commands"]:
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=False)
    record = output / "variant.json"
    plan["input_sha256"] = {str(path): digest(path) for path in inputs}
    record.write_text(json.dumps(plan, indent=2) + "\n")
    try:
        for source, relative, expected in files:
            destination = output / "baseline-kernels" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            require(digest(destination) == expected, "baseline changed during copy: " + str(source))
        # Keep original graph/reference paths for provenance resolution. These
        # links are read-only inputs by contract; compiler outputs go elsewhere.
        for name in ("replacement", "nr-prefill", "nr-decode", "quant-nr-trace"):
            (output / name).symlink_to(os.path.relpath(stage / name, output), target_is_directory=True)
        for index, command in enumerate(plan["commands"], 1):
            with (output / f"{index:02d}-build.log").open("w") as log:
                subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        require(all(digest(Path(path)) == expected for path, expected in plan["input_sha256"].items()),
                "immutable graph/reference/resource inputs changed during the build")
        plan["status"] = "BUILT_PREPARED_NOT_BOARD_VALIDATED"
        plan["image_sha256"] = digest(output / "image/qwen_model.bin")
        plan["archive_sha256"] = digest(output / "model-lib/libqwen_triton.a")
    except Exception as error:
        plan["status"] = "FAILED_DO_NOT_DEPLOY"
        plan["error"] = str(error)
        raise
    finally:
        record.write_text(json.dumps(plan, indent=2) + "\n")
    print(output / "prepared/ddr-load.plan")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
