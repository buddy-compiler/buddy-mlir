#!/usr/bin/env python3
"""Archive this experiment's derived ELF, terminal run, and inherited oracle.

Run with the qwen3fpga-py311 Python environment. This does not synthesize an
image.json for the patched image, change production archivers, or fetch remote
files. A terminal failed/timeout run is retained as NOT_ACCEPTED.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib


MODEL = Path(__file__).resolve().parents[2]
ROOT = MODEL.parents[3]
TOOLS = MODEL / "tools"
sys.path.insert(0, str(TOOLS))
import archive_model_run as existing
import prepare_startup_ab as startup
from prepare_model_run import check_image_bytes, digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def artifact(path):
    require(path.is_file() and not path.is_symlink(), f"missing/symlink evidence: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)}


def elapsed_seconds(value):
    parts = value.split(":")
    total = 0.0
    for part in parts:
        total = total * 60 + float(part)
    return total


def timing(raw):
    result = {"scope": "local host orchestration process, including transfer/wait/collection",
              "wall_seconds": None, "host_user_cpu_seconds": None,
              "host_system_cpu_seconds": None, "raw_text": raw,
              "limitation": "Host CPU time is not RA CPU time and does not establish a hardware wait."}
    for label, value in re.findall(r"([A-Za-z_]+)\s*=\s*([0-9:.]+)", raw):
        label = label.lower()
        key = ("wall_seconds" if label in ("wall", "wall_seconds", "elapsed", "elapsed_seconds", "real")
               else "host_user_cpu_seconds" if label in ("user", "user_seconds", "user_cpu_seconds")
               else "host_system_cpu_seconds" if label in ("sys", "system", "sys_seconds", "system_seconds")
               else None)
        if key:
            result[key] = elapsed_seconds(value)
    for label, key in ((r"User time \(seconds\)", "host_user_cpu_seconds"),
                       (r"System time \(seconds\)", "host_system_cpu_seconds"),
                       (r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\)", "wall_seconds")):
        match = re.search(label + r":\s*([0-9:.]+)", raw)
        if match:
            result[key] = elapsed_seconds(match[1])
    return result


def remote_location(stderr, run_id):
    matches = re.findall(r"^\[fpga_run\] Remote (?:logs|files): (\S+)\s*$", stderr, re.M)
    if matches:
        require(all(v.endswith("/fpga-runs/" + run_id) for v in matches), "remote run ID mismatch")
        return matches[-1]
    matches = re.findall(r"^\[fpga_run\] Server workdir: (\S+)\s*$", stderr, re.M)
    require(len(set(matches)) == 1, "missing/ambiguous remote server workdir evidence")
    return matches[0].rstrip("/") + "/fpga-runs/" + run_id


def graph_progress(raw):
    text = raw.decode(errors="replace")

    def stages(pattern):
        return [{"kind": kind, "position": int(position, 16),
                 "position_hex": "0x" + position.lower()}
                for kind, position in re.findall(pattern, text)]

    starts = stages(r"\[model\] (prefill|decode) begin position=([0-9A-Fa-f]+)")
    completed = stages(r"\[model\] (prefill|decode) position=([0-9A-Fa-f]+) token=")
    return {"graph_starts_observed": len(starts), "graph_completions_observed": len(completed),
            "last_graph_started": starts[-1] if starts else None,
            "last_graph_completed": completed[-1] if completed else None,
            "started_graph_without_completion": starts[-1] if len(starts) > len(completed) else None,
            "scope": "UART graph boundaries only; this is not a sampled RA program counter"}


def run_checker(command, output, log):
    completed = subprocess.run([str(v) for v in command], capture_output=True, text=True)
    log.write_text(completed.stdout + completed.stderr)
    result = read(output) if output.is_file() else {"status": "NOT_ACCEPTED", "error": "checker produced no JSON"}
    if not output.is_file():
        output.write_text(json.dumps(result, indent=2) + "\n")
    return {"argv": [str(v) for v in command], "returncode": completed.returncode,
            "status": result.get("status", "NOT_ACCEPTED"), "report": output.name, "log": log.name}, result


def archive(args):
    output = args.output.absolute()
    require(not output.exists() and not output.is_symlink(), "output must be a new directory")
    variant, run = args.variant_dir.resolve(strict=True), args.run_dir.resolve(strict=True)
    require(re.fullmatch(r"run-[0-9a-f]{16}", run.name) is not None, "invalid run directory name")
    image, prepared = variant / "image", variant / "prepared"
    # result.json is written atomically after the worker has ended. Never
    # archive an actively growing run as a completed experiment.
    result = read(run / "result.json")
    require(result.get("status") in ("OK", "ERROR", "TIMEOUT", "CANCELLED", "STOPPED"),
            "run does not have a recognized terminal result")
    raw_uart = (run / "uart.raw.log").read_bytes()
    uart_hash = hashlib.sha256(raw_uart).hexdigest()
    patch = read(image / "startup-ab.json")
    source_elf = Path(patch["source_elf"])
    original, actual = source_elf.read_bytes(), (image / "qwen_model.elf").read_bytes()
    reconstructed, fresh_patch = startup.patch_elf(original, patch["variant"])
    require(reconstructed == actual, "derived ELF differs from independently reapplied exact patch")
    for key in ("source_elf_sha256", "elf_sha256", "patches", "byte_differences", "invariants", "graph_completion_calls"):
        require(patch.get(key) == fresh_patch[key], "patch manifest mismatch: " + key)
    require(patch.get("status") == "PASS" and patch.get("undefined_symbols") == [], "patch audit did not pass")
    require(patch.get("tool_sha256") == digest(TOOLS / "prepare_startup_ab.py"), "patch tool identity changed")
    require(patch.get("bin_sha256") == digest(image / "qwen_model.bin"), "derived raw BIN hash mismatch")
    elf_hash = hashlib.sha256(actual).hexdigest()
    audit = read(image / "elf-audit.json")
    require(audit.get("status") == "PASS" and audit.get("elf_sha256") == elf_hash, "derived ELF audit mismatch")
    prime = source_elf.parent
    plan, build_record = read(prime / "w8a8-image-plan.json"), read(prime / "image.json")
    reference = read(prime / "numeric-reference.json")
    require(plan.get("ame_startup") == "prime" and plan.get("layers") == 28,
            "source-prime workload identity mismatch")
    require(plan.get("numeric_reference") == reference, "inherited plan/reference mismatch")
    arrays, metadata = Path(reference["source"]), Path(reference["metadata_source"])
    require(digest(metadata) == reference["metadata_sha256"], "reference metadata hash mismatch")
    oracle_tokens = existing.verify_reference(reference, read(metadata), arrays,
        prime / "numeric-reference.bin", layers=28, prefill=16, steps=8)
    symbols = startup.Elf(actual).symbols
    oracle_address = symbols["model_reference_raw"][0][3]
    oracle_offset = existing.elf_blob_matches(image / "qwen_model.elf", oracle_address,
                                            prime / "numeric-reference.bin")
    deployment = read(prepared / "deployment.json")
    uploaded = read(run / "run-manifest.json")
    ddr_plan = tomllib.loads((prepared / "ddr-load.plan").read_text())
    segments = ddr_plan["segments"]
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    check(deployment.get("elf_sha256") == elf_hash, "deployment ELF hash mismatch")
    check(uploaded.get("plan_sha256") == digest(prepared / "ddr-load.plan"), "uploaded DDR plan mismatch")
    check(uploaded.get("segments") == [{k: s[k] for k in ("file", "size", "sha256")} for s in segments],
          "uploaded segment manifest mismatch")
    check(segments == [{k: v for k, v in s.items() if k != "arena_end"} for s in deployment["segments"]],
          "prepared plan/deployment segments mismatch")
    check(result.get("uart_bytes") == len(raw_uart), "worker UART length mismatch")
    check(result.get("fpga") == 5, "expected FPGA5")
    model_segments = [s for s in segments if s["name"] == "model"]
    require(len(model_segments) == 1, "missing/duplicate model DDR segment")
    model_segment = model_segments[0]
    check(result.get("sha256") == model_segment["sha256"], "worker model image identity mismatch")
    stderr = (variant / "runner.stderr.log").read_text(errors="replace")
    remote = remote_location(stderr, run.name)
    resources = []
    for segment in segments:
        name = segment["file"]
        require(Path(name).name == name and name not in (".", ".."), "unsafe segment filename")
        identity = artifact(prepared / name)
        check(identity["bytes"] == segment["size"] and identity["sha256"] == segment["sha256"],
              "local prepared segment differs from uploaded identity: " + name)
        resources.append({"name": segment["name"], "prepared_local": identity,
                          "remote_file": remote + "/" + name,
                          "remote_readback": remote + "/" + name + ".readback",
                          "expected_readback_sha256": segment["sha256"],
                          "worker_readback_match_reported": result.get("segment_readbacks", {}).get(name),
                          "readback_downloaded_by_this_archiver": False})
    try:
        _, _, _, verified_readbacks = existing.verify_run_identity(run, prepared, elf_hash, raw_uart)
        run_identity = {"status": "PASS", "readbacks": verified_readbacks}
    except (ValueError, KeyError, OSError) as error:
        run_identity = {"status": "NOT_ACCEPTED", "error": str(error)}
        failures.append("completed-run identity validation: " + str(error))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".derived-run-archive-", dir=output.parent) as temporary:
        stage = Path(temporary)
        copied = {}

        def copy(source, target):
            before = artifact(source)
            dest = stage / target
            dest.parent.mkdir(parents=True, exist_ok=True)
            require(not dest.exists(), "archive path collision: " + str(target))
            shutil.copyfile(source, dest)
            require(digest(dest) == before["sha256"], "evidence changed while copying: " + str(source))
            copied[str(target)] = before

        for name in ("qwen_model.elf", "qwen_model.bin", "elf-audit.json", "startup-ab.json"):
            copy(image / name, Path("derived-image") / name)
        for name in ("qwen_model.elf", "image.json", "w8a8-image-plan.json", "numeric-reference.json", "model_main.c"):
            copy(prime / name, Path("source-prime") / name)
        for name in ("result.json", "run-manifest.json", "uart.raw.log", "worker.log", "uvhs.log"):
            if (run / name).is_file():
                copy(run / name, Path("run") / name)
            else:
                failures.append("missing terminal-run evidence: " + name)
        for name in ("deployment.json", "ddr-load.plan"):
            copy(prepared / name, Path("deployment") / name)
        for name in ("host-time.txt", "runner.stdout.log", "runner.stderr.log"):
            copy(variant / name, Path("host") / name)
        time_report = timing((stage / "host/host-time.txt").read_text(errors="replace"))
        check(bool(time_report["raw_text"].strip()), "host timing record is empty; local runner may not have ended")
        extract = stage / "objcopy-check.bin"
        llvm = ROOT / "llvm/build-2d26/bin"
        subprocess.run([str(llvm / "llvm-objcopy"), "-O", "binary",
                        str(stage / "derived-image/qwen_model.elf"), str(extract)], check=True)
        check_image_bytes(stage / "derived-image/qwen_model.bin", extract)
        check_image_bytes(prepared / model_segment["file"], extract)
        check(deployment.get("elf_objcopy_bytes") == extract.stat().st_size, "objcopy length identity mismatch")
        extract.unlink()
        host_arrays = MODEL / "build/quant-opt/shared/combined-host-run/arrays.npz"
        numeric_call, numeric = run_checker([sys.executable, TOOLS / "check_board_trace.py",
            "--uart", stage / "run/uart.raw.log", "--host-graph", host_arrays,
            "--quant-reference", arrays, "--embedded-reference", stage / "source-prime/numeric-reference.json",
            "--layers", "28", "--prefill", "16", "--steps", "8", "--output", stage / "numeric-verification.json"],
            stage / "numeric-verification.json", stage / "numeric-verification.validator.log")
        text_call, text = run_checker([sys.executable, TOOLS / "check_fixed_text.py",
            "--uart", stage / "run/uart.raw.log", "--image-plan", stage / "source-prime/w8a8-image-plan.json",
            "--assets", MODEL / "assets/official", "--output", stage / "fixed-text-verification.json"],
            stage / "fixed-text-verification.json", stage / "fixed-text-verification.validator.log")
        full = numeric.get("full_tensor_fpga_vs_quantized_reference", {})
        checks = full.get("checks", [])
        check(numeric_call["returncode"] == 0 and numeric.get("status") == "FULL_LOGITS_KV_PASS"
              and len(checks) == 27 and all(row.get("firmware_verdict") == "PASS" for row in checks),
              "27 full logits/KV checks did not pass")
        check(text_call["returncode"] == 0 and text.get("status") == "FIXED_TEXT_PASS"
              and [r["token"] for r in text.get("predictions", [])] == oracle_tokens,
              "fixed prompt/tokenizer/nine-token trajectory did not pass")
        check(result.get("status") == "OK" and result.get("completion_marker_seen") is True,
              "worker did not report successful completion")
        source_inputs = []
        for name, expected in build_record.get("input_sha256", {}).items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            current = digest(path) if path.is_file() else None
            source_inputs.append({"path": str(path), "recorded_source_prime_sha256": expected,
                                  "current_sha256": current, "current_matches_source_prime": current == expected})
        for name in ("prepare_startup_ab.py", "archive_model_run.py", "prepare_model_run.py",
                     "check_board_trace.py", "check_fixed_text.py", "check_kernel_profile.py"):
            copy(TOOLS / name, Path("validators") / name)
        copy(Path(__file__), Path("validators/archive_derived_run.py"))
        copy(metadata, Path("references/quant-reference.json"))
        require(digest(run / "uart.raw.log") == uart_hash, "terminal UART changed during archive")
        require(digest(run / "result.json") == copied["run/result.json"]["sha256"], "terminal result changed during archive")
        report = {"schema_version": 1,
            "status": "DERIVED_STARTUP_RUN_PASS" if not failures else "NOT_ACCEPTED",
            "created_utc": datetime.now(timezone.utc).isoformat(), "run_id": run.name,
            "fpga": result.get("fpga"), "variant": patch["variant"], "failures": failures,
            "derivation": {"method": "exact ELF instruction replacement; no recompilation",
                           "source_prime_elf_sha256": startup.SOURCE_SHA256,
                           "derived_elf_sha256": elf_hash, "patch_manifest": "derived-image/startup-ab.json",
                           "independently_reapplied_patch_matches": True,
                           "source_build_record": "source-prime/image.json",
                           "source_plan_role": "inherited workload/oracle contract; ame_startup remains prime in original metadata"},
            "identity": {"derived_raw_bin": artifact(image / "qwen_model.bin"),
                         "prepared_boot_image": artifact(prepared / model_segment["file"]),
                         "raw_and_canonically_zero_padded_boot_match_derived_elf": True,
                         "raw_uart_sha256": uart_hash, "run_identity": run_identity,
                         "resources": resources, "source_prime_inputs": source_inputs},
            "inherited_oracle": {"manifest": artifact(prime / "numeric-reference.json"),
                                 "plan": artifact(prime / "w8a8-image-plan.json"),
                                 "blob": artifact(prime / "numeric-reference.bin"),
                                 "arrays": artifact(arrays), "metadata": artifact(metadata),
                                 "compiled_host_arrays": artifact(host_arrays),
                                 "address": hex(oracle_address), "elf_offset": hex(oracle_offset),
                                 "npz_repacked_and_embedded_bytes_match": True,
                                 "expected_nine_tokens": oracle_tokens},
            "validation": {"numeric": numeric_call, "fixed_text": text_call,
                           "full_tensor_check_count": len(checks),
                           "expected_full_tensor_check_count": 27,
                           "full_tensor_firmware_pass_count": sum(row.get("firmware_verdict") == "PASS"
                                                                  for row in checks)},
            "progress": graph_progress(raw_uart),
            "time": time_report,
            "worker_capture_limit_seconds": result.get("capture_seconds"),
            "last_uart_lines": raw_uart.decode(errors="replace").splitlines()[-8:],
            "limits": ["Source-prime build metadata describes the source ELF, not a recompiled derived build.",
                       "Full last-position logits and valid KV only; no hidden-state acceptance.",
                       "Remote readback hashes are expected identities with worker match attestations; this script downloads nothing.",
                       "Host CPU and wall time include orchestration; neither supplies a stalled RA PC."],
            "archived_sources": copied}
        report["archive_sha256"] = {str(p.relative_to(stage)): digest(p)
                                    for p in sorted(stage.rglob("*")) if p.is_file()}
        (stage / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        require(not output.exists() and not output.is_symlink(), "archive output appeared during validation")
        stage.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("variant-dir", "run-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        report = archive(args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print("derived run archive rejected: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "run_id": report["run_id"], "output": str(args.output)}, indent=2))
    return 0 if report["status"] == "DERIVED_STARTUP_RUN_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
