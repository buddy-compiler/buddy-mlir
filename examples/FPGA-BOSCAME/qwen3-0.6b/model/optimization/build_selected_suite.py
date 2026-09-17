#!/usr/bin/env python3
"""Link the exact 18 optimized model kernels into a standalone NR oracle suite.

Read variant.json and its immutable model archive to select 11 INT8 linears and
7 dequantizers. Snapshot the recorded objects without rebuilding kernels. The
normal Triton builder supplies the launchers, shared NR runtime and suite link.
No FPGA action is performed. Existing output directories are refused.
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import sys

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent
QWEN = MODEL.parent
REPO = MODEL.parents[3]
sys.path.insert(0, str(HERE))
from build_dequant_ab import inspect_case, sha
sys.path.insert(0, str(MODEL / "tools"))
from build_nr_w8a8_image import resolve_linker


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(variant_path, output, linker):
    variant_path = variant_path.resolve()
    output = output.absolute()
    require(not output.exists() and not output.is_symlink(),
            "output exists; use a fresh suite directory")
    variant = json.loads(variant_path.read_text())
    require(variant.get("status") == "BUILT_PREPARED_NOT_BOARD_VALIDATED",
            "variant must contain a successfully built/prepared model")
    archive_dir = variant_path.parent / "model-lib"
    archive_path = archive_dir / "archive.json"
    archive = json.loads(archive_path.read_text())
    require(sha(archive_dir / "libqwen_triton.a") == archive["sha256"] == variant["archive_sha256"],
            "variant/archive library hash mismatch")
    records = {case["case"]: case for case in archive["cases"]
               if case["frontend"]["family"] in ("matmul_i8", "dequantization")}
    counts = {family: sum(r["frontend"]["family"] == family for r in records.values())
              for family in ("matmul_i8", "dequantization")}
    require(counts == {"matmul_i8": 11, "dequantization": 7},
            "expected exactly 11 model INT8 linears and 7 dequantizers")
    replacements = {Path(path).name: Path(path).resolve() for path in variant["replace_cases"]}
    require(len(replacements) == len(variant["replace_cases"]) and set(replacements) == set(records),
            "variant replacements must be exactly those 18 semantic model cases")
    cases, inputs, snapshots = [], {variant_path: sha(variant_path), archive_path: sha(archive_path)}, []
    case_details = {}
    for name, record in sorted(records.items()):
        source = replacements[name]
        family = record["frontend"]["family"]
        frontend, _ = inspect_case(source, "ame" if family == "matmul_i8" else "dequant")
        require(frontend == record["frontend"], "replacement frontend differs from model archive: " + name)
        for filename, expected in record["files_sha256"].items():
            native = filename in ("kernel.o", "adapter.o", "kernel.ll", "kernel.s", "kernel.nr.S")
            relative = Path("nr" if native else "") / filename
            artifact = source / relative
            require(sha(artifact) == expected, "replacement changed since model archive: " + str(artifact))
            inputs[artifact] = expected
            snapshots.append((artifact, Path(name) / relative, expected))
        native_manifest = source / "nr/manifest.json"
        native = json.loads(native_manifest.read_text())
        original_native = archive_dir / "evidence" / name / "manifest.json"
        require(sha(native_manifest) == sha(original_native), "replacement native configuration changed: " + name)
        inputs[native_manifest] = sha(native_manifest)
        snapshots.append((native_manifest, Path(name) / "nr/manifest.json", sha(native_manifest)))
        launch = Path(frontend["directory"]) / "launch.c"
        metadata = launch.parent / "metadata.json"
        require(launch.is_file() and sha(launch) == native["launch_sha256"],
                "parent launch differs from the compiled case oracle: " + name)
        require(metadata.is_file(), "missing parent case metadata: " + name)
        inputs[launch] = sha(launch)
        inputs[metadata] = sha(metadata)
        cases.append(frontend)
        case_details[name] = {"directory": str(source), "family": family,
                              "kernel_sha256": record["files_sha256"]["kernel.o"],
                              "adapter_sha256": record["files_sha256"]["adapter.o"],
                              "native_manifest_sha256": sha(native_manifest),
                              "ame_gpr_mode": native.get("ame_gpr_mode", "fixed"),
                              "nr_coalesce_fences": native.get("nr_coalesce_fences", False),
                              "constexprs": frontend["constexprs"], "grid": frontend["grid"]}
    resolved_linker = resolve_linker(linker, REPO / "llvm/build-2d26/bin")
    # Existing build_suite consumes precompiled objects and the original oracle
    # launchers. Give it copied case trees and a separate runtime output path.
    sys.path.insert(0, str(QWEN / "triton"))
    import build as triton_build
    flags = [f"RISCV_LD={resolved_linker['path']}", f"BUILD={output / 'runtime'}",
             "AME_GPR_MODE=direct", "NR_COALESCE_FENCES=1"]
    config = triton_build.get_config(cases[0], flags)
    runtime = [Path(cases[0]["directory"]) / obj
               for obj in shlex.split(config["variables"]["RUNTIME_OBJS"])]
    for path in triton_build.source_provenance(cases, False):
        original = REPO / path
        inputs[original] = sha(original)
    plan = {"status": "PLANNED_NOT_BUILT", "variant": str(variant_path),
            "variant_sha256": sha(variant_path), "model_archive_sha256": archive["sha256"],
            "case_count": len(cases), "family_counts": counts, "cases": case_details,
            "linker": resolved_linker, "build_configuration": config,
            "scope": "standalone parent-oracle kernel suite; not model inference or model validation",
            "compiler_policy": "reuse recorded kernel/adapter objects; compile launchers and shared runtime only",
            "input_sha256": {str(path): expected for path, expected in sorted(inputs.items())},
            "runtime_build_command": ["make", "--no-print-directory", "-s", "-C", cases[0]["directory"],
                                      *map(str, runtime), *flags]}
    return output, plan, cases, snapshots, runtime, triton_build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--linker", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output, plan, cases, snapshots, runtime, triton_build = prepare(args.variant, args.output, args.linker)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    record = output / "selected-suite.json"
    record.write_text(json.dumps(plan, indent=2) + "\n")
    try:
        root = output / "kernels"
        for source, relative, expected in snapshots:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            require(sha(destination) == expected, "input changed while snapshotting: " + str(source))
        triton_build.command(plan["runtime_build_command"], output=output / "runtime-build.log")
        plan["runtime_objects_sha256"] = {str(p.relative_to(output)): sha(p) for p in runtime}
        triton_build.BUILD_ROOT = root
        suite = triton_build.build_suite(cases, plan["build_configuration"], False, runtime, "optimized-model-kernels")
        suite_dir = root / "suite-optimized-model-kernels"
        object_hashes = {str(root / case["name"] / "nr" / filename): sha(root / case["name"] / "nr" / filename)
                         for case in cases for filename in ("kernel.o", "adapter.o")}
        object_hashes.update({str(path): sha(path) for path in runtime})
        sidecar = {"status": "BUILT_AUDITED_NOT_BOARD_RUN", "scope": plan["scope"],
                   "case_count": len(cases), "case_order": [case["name"] for case in cases],
                   "image_sha256": sha(suite_dir / "suite.bin"),
                   "elf_sha256": sha(suite_dir / "suite.elf"),
                   "input_objects_sha256": object_hashes,
                   "suite_objects_sha256": {path.name: sha(path) for path in sorted(suite_dir.glob("*.o"))},
                   "suite_source_sha256": sha(suite_dir / "suite.c"),
                   "suite_manifest_sha256": sha(suite_dir / "manifest.json"),
                   "elf_audit": suite["elf_audit"], "cases": {}}
        for case in cases:
            name = case["name"]
            details = plan["cases"][name]
            sidecar["cases"][name] = {
                "source_directory": str(root / name),
                "original_directory": details["directory"],
                "frontend_manifest_sha256": sha(root / name / "frontend.json"),
                "native_manifest_sha256": sha(root / name / "nr/manifest.json"),
                "elf_audit": "PASS", "ame_gpr_mode": details["ame_gpr_mode"],
                "nr_coalesce_fences": details["nr_coalesce_fences"],
                "kernel_module": case.get("kernel_module", "kernels")}
        (suite_dir / "object-provenance.json").write_text(json.dumps(sidecar, indent=2) + "\n")
        require(all(sha(Path(path)) == expected for path, expected in plan["input_sha256"].items()),
                "original input changed during suite build")
        plan.update(status="BUILT_NOT_BOARD_VALIDATED", suite=suite,
                    suite_image=str(root / "suite-optimized-model-kernels/suite.bin"))
    except Exception as error:
        plan.update(status="FAILED_DO_NOT_DEPLOY", error=str(error))
        raise
    finally:
        record.write_text(json.dumps(plan, indent=2) + "\n")
    print(plan["suite_image"])


if __name__ == "__main__":
    main()
