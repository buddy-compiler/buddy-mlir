#!/usr/bin/env python3
"""Require all 18 selected model kernel checks, then archive compact evidence."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dequant_ab import inspect_case, sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_uart(text, names):
    begins = re.findall(r"^\[(\d+)/(\d+)\] Triton (\w+) BEGIN\r?$", text, re.M)
    require(begins == [(str(i), str(len(names)), name) for i, name in enumerate(names, 1)],
            "missing, duplicate, or reordered case starts")
    checks = re.findall(r"^verify ([\w ]+): (PASS|FAIL) errors=([0-9a-fA-F]{8}) "
                        r"max_abs_error_f32_bits=([0-9a-fA-F]{8})\r?$", text, re.M)
    require(checks == [(name, "PASS", "00000000", "00000000")
                       for name in [*names, "qwen3 Triton operator suite"]],
            "every case and suite must have exactly one zero-error PASS")
    require(text.count("[nr] RA returned: PASS") == 1 and
            text.count("verify NR runtime: PASS") == 1 and "FAIL" not in text,
            "missing/failed NR return")
    cycles = re.findall(r"^cycles (\w+): ([0-9a-fA-F]{16})\r?$", text, re.M)
    expected = [name for name in names if name.startswith("matmul_")]
    require([name for name, _ in cycles] == expected and all(int(value, 16) > 0 for _, value in cycles),
            "missing or reordered matmul cycle samples")
    launch = re.findall(r"\[nr\] launch cycles=0x([0-9a-fA-F]+) status=0x([0-9a-fA-F]+)", text)
    require(len(launch) == 1 and int(launch[0][1], 16) == 0, "missing successful launch status")
    return {"case_count": len(names), "cases": {name: {"status": "PASS", "errors": 0, "max_abs": 0}
                                                 for name in names},
            "matmul_cycles": {name: int(value, 16) for name, value in cycles},
            "launch_cycles": int(launch[0][0], 16)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.suite / "manifest.json"
    provenance_path = args.suite / "object-provenance.json"
    manifest, provenance = [json.loads(p.read_text()) for p in (manifest_path, provenance_path)]
    runner = json.loads((args.run / "result.json").read_text())
    names = manifest["cases"]
    require(len(names) == len(set(names)) == 18 and names == provenance["case_order"],
            "suite/provenance must contain the same 18 distinct cases")
    require(sum(name.startswith("matmul_") for name in names) == 11 and
            sum(name.startswith("dequantize_") for name in names) == 7,
            "suite must contain 11 linears and 7 dequantizers")
    require(sha(args.suite / "suite.bin") == manifest["sha256"] == provenance["image_sha256"] == runner["sha256"],
            "image/build/provenance/upload hash mismatch")
    require(sha(manifest_path) == provenance["suite_manifest_sha256"] and
            sha(args.suite / "suite.elf") == provenance["elf_sha256"] == manifest["elf_audit"]["elf_sha256"],
            "manifest/ELF audit hash mismatch")
    require(manifest["elf_audit"]["status"] == "PASS" and provenance["elf_audit"]["status"] == "PASS",
            "missing successful ELF audit")
    require(runner["status"] == "OK" and runner["ddr_readback_matches"] is True and runner["completion_marker_seen"] is True,
            "upload/readback/completion failed")
    uart = args.run / "uart.raw.log"
    require(uart.stat().st_size == runner["uart_bytes"], "UART length differs from runner")
    result = check_uart(uart.read_text(errors="replace"), names)
    for path, expected in provenance["input_objects_sha256"].items():
        require(sha(Path(path)) == expected, "input object changed: " + path)
    for name, expected in provenance["suite_objects_sha256"].items():
        require(sha(args.suite / name) == expected, "suite object changed: " + name)
    require(sha(args.suite / "suite.c") == provenance["suite_source_sha256"], "suite launcher changed")
    for name, details in provenance["cases"].items():
        source = Path(details["source_directory"])
        front, _ = inspect_case(source, "ame" if name.startswith("matmul_") else "dequant")
        require(front["name"] == name and sha(source / "frontend.json") == details["frontend_manifest_sha256"] and
                sha(source / "nr/manifest.json") == details["native_manifest_sha256"],
                "case source/configuration changed: " + name)
    require(not args.output.exists(), "archive exists; choose a fresh output")
    args.output.mkdir(parents=True)
    for source, filename in ((manifest_path, "manifest.json"), (provenance_path, "object-provenance.json"),
                             (uart, "uart.raw.log"), (args.run / "result.json", "runner-result.json")):
        shutil.copy2(source, args.output / filename)
    (args.output / "elf-audit.json").write_text(json.dumps(manifest["elf_audit"], indent=2) + "\n")
    result.update(status="PASS", run_id=args.run.name, image_sha256=manifest["sha256"],
                  uart_sha256=sha(uart), manifest_sha256=sha(manifest_path),
                  provenance_sha256=sha(provenance_path), ddr_readback_matches=True,
                  scope="18 standalone optimized model kernel shapes; not a model inference result")
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "run_id": args.run.name, "cases": len(names),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
