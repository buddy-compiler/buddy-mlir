#!/usr/bin/env python3
"""Strictly score and archive one runtime-length attention Stage A FPGA run."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = args.build / "suite-all"
    manifest = json.loads((suite / "manifest.json").read_text())
    run = json.loads((args.run / "result.json").read_text())
    if (run.get("status") != "OK" or run.get("ddr_readback_matches") is not True
            or run.get("completion_marker_seen") is not True
            or run.get("sha256") != digest(suite / "suite.bin")
            or manifest["sha256"] != run["sha256"]):
        raise ValueError("run did not complete with this exact image and DDR readback")
    raw = (args.run / "uart.raw.log").read_text()
    text = raw.replace("\\r\\n", "\n").replace("\r", "")
    starts = list(re.finditer(r"\[(\d+)/(\d+)\] Triton (\w+) BEGIN", text))
    names = manifest["cases"]
    if len(starts) != len(names) or [m[3] for m in starts] != names:
        raise ValueError("missing, duplicate or reordered case header")
    report = {"status": "PASS", "scope": "Stage A QK/PV kernels only, not model performance",
              "run_id": args.run.name, "fpga": run["fpga"], "image_sha256": run["sha256"],
              "cases": {}, "total_length_checks": 0}
    for i, (name, start) in enumerate(zip(names, starts)):
        if int(start[1]) != i + 1 or int(start[2]) != len(names):
            raise ValueError("wrong suite ordinal")
        part = text[start.end():starts[i+1].start() if i+1 < len(starts) else len(text)]
        matches = re.findall(r"\[attention-position\] valid=([0-9a-fA-F]+) errors=([0-9a-fA-F]+) kernel_cycles=([0-9a-fA-F]+)", part)
        records = [dict(valid=int(a,16), errors=int(b,16), kernel_cycles=int(c,16)) for a,b,c in matches]
        metadata = json.loads((args.cases / name / "metadata.json").read_text())
        if ([r["valid"] for r in records] != metadata["valid_lengths"]
                or any(r["errors"] or r["kernel_cycles"] <= 0 for r in records)
                or part.count("[attention-position]") != len(records)):
            raise ValueError("missing/extra lengths, numeric errors or invalid cycle counts: " + name)
        expected = f"verify {name}: PASS errors=00000000 max_abs_error_f32_bits=00000000"
        if part.count(expected) != 1 or "FAIL" in part:
            raise ValueError("case did not report exactly one numerical PASS")
        by_length = {length: [r["kernel_cycles"] for r in records if r["valid"] == length]
                     for length in (16,512)}
        ratio = ((sum(by_length[512])/len(by_length[512])) /
                 (sum(by_length[16])/len(by_length[16]))
                 if by_length[512] and by_length[16] else None)
        report["cases"][name] = {"length_checks": records,
            "kernel_cycles_512_over_16": ratio,
            "performance_scope": "same dynamic kernel, valid512 versus valid16; mean repeated lengths; excludes oracle/UART and is not old-kernel speedup"}
        report["total_length_checks"] += len(records)
    for marker in ("verify qwen3 Triton operator suite: PASS errors=00000000",
                   "[nr] RA returned: PASS", "verify NR runtime: PASS"):
        if text.count(marker) != 1:
            raise ValueError("missing/duplicate completion: " + marker)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("uart.raw.log", "uvhs.log", "worker.log", "result.json"):
        shutil.copyfile(args.run / name, args.output / name)
    shutil.copyfile(suite / "manifest.json", args.output / "suite-manifest.json")
    shutil.copyfile(suite / "suite.audit.json", args.output / "suite.audit.json")
    for name in names:
        destination = args.output / "cases" / name
        destination.mkdir(parents=True)
        for filename in ("frontend.json", "adapter.c", "kernel.ttir", "kernel.linalg.mlir"):
            shutil.copyfile(args.build/name/filename, destination/filename)
        for filename in ("metadata.json", "launch.c"):
            shutil.copyfile(args.cases/name/filename, destination/filename)
        shutil.copyfile(args.build/name/"nr/manifest.json", destination/"nr-manifest.json")
        shutil.copyfile(args.build/name/"nr/kernel.s", destination/"kernel.s")
        shutil.copyfile(args.build/name/"host/output.log", destination/"host.log")
        shutil.copyfile(args.build/name/"nr/vector-host.log", destination/"vector-host.log")
    # Snapshot actual sources as well as their build-time manifest digests.
    repo = Path(__file__).resolve().parents[5]
    for relative, expected in manifest["source_files"].items():
        source = repo / relative
        if source.is_file() and digest(source) == expected:
            destination = args.output / "sources" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    report["archive_sha256"] = {str(p.relative_to(args.output)): digest(p)
                               for p in sorted(args.output.rglob("*")) if p.is_file()}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "length_checks": report["total_length_checks"],
        "kernel_cycles_512_over_16": {n:r["kernel_cycles_512_over_16"] for n,r in report["cases"].items()}}, indent=2))


if __name__ == "__main__":
    main()
