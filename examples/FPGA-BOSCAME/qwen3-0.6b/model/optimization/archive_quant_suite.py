#!/usr/bin/env python3
"""Archive the six model quantizers after exact image/IR/UART checks."""
import argparse
import json
from pathlib import Path
import shutil
from archive_selected_suite import check_uart, require
from build_dequant_ab import inspect_case, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("build", "run", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    suite = args.build / "suite-all"
    manifest = json.loads((suite / "manifest.json").read_text())
    result = json.loads((args.run / "result.json").read_text())
    names = [f"quantize_{m}x{k}" for m in (16, 1) for k in (1024, 2048, 3072)]
    require(manifest["cases"] == names and manifest["case_count"] == 6 and not manifest["host"], "wrong suite")
    require(sha(suite / "suite.bin") == manifest["sha256"] == result["sha256"], "image hash mismatch")
    require(manifest["elf_audit"]["status"] == "PASS" and
            sha(suite / "suite.elf") == manifest["elf_audit"]["elf_sha256"], "ELF audit mismatch")
    require(result["status"] == "OK" and result.get("ddr_readback_matches") is True and
            result.get("completion_marker_seen") is True, "incomplete run/readback")
    uart = args.run / "uart.raw.log"
    require(uart.stat().st_size == result["uart_bytes"], "UART length mismatch")
    report = check_uart(uart.read_text(), names)
    inputs = {}
    for name in names:
        case = args.build / name
        front, _ = inspect_case(case, "quant")
        require(front.get("quantization_lowering") == "rvv", "not RVV quantization")
        for stage, path in (("frontend", "frontend.json"), ("adapter", "adapter.c"),
                            ("ttir", "kernel.ttir"), ("linalg", "kernel.linalg.mlir"), ("llvm", "nr/kernel.ll")):
            require(sha(case / path) == manifest["kernels"][name][stage], "stale kernel IR: " + name)
        inputs[name] = {path: sha(case / path) for path in
                       ("nr/kernel.o", "nr/adapter.o", "nr/kernel.nr.S", "nr/manifest.json")}
    require(not args.output.exists(), "archive already exists")
    args.output.mkdir(parents=True)
    for source, filename in ((uart, "uart.raw.log"), (args.run / "result.json", "runner-result.json"),
                             (suite / "manifest.json", "build-manifest.json")):
        shutil.copy2(source, args.output / filename)
    for name in names:
        shutil.copy2(args.build / name / "nr/manifest.json", args.output / (name + "-native-manifest.json"))
    report.update(status="PASS", run_id=args.run.name, image_sha256=manifest["sha256"],
                  uart_sha256=sha(uart), input_hashes=inputs,
                  scope="six standalone RVV quantizers; full-model acceptance is separate")
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(names)} quantizers -> {args.output}")


if __name__ == "__main__":
    main()
