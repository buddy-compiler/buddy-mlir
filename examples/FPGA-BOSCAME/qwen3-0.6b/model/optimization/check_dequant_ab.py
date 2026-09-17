#!/usr/bin/env python3
"""Validate complete A/B UART evidence and summarize independently timed calls."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import struct


def fields(line):
    return dict(re.findall(r"(\w+)=(\S+)", line))


def parse(text, *, platform, manifest=None):
    lines = [line.strip() for line in text.splitlines() if "[dequant-ab]" in line]
    configs = [line for line in lines if line.startswith("[dequant-ab] config ")]
    if len(configs) != 1:
        raise ValueError("expected one config record; reject partial/concatenated runs")
    config = fields(configs[0])
    shape = [int(config[k], 16) for k in ("rows", "cols")]
    repeats = int(config["repeats"], 16)
    if min(shape) < 1 or repeats < 2 or config["offset_words"] != "00000010":
        raise ValueError("invalid shape, repeat count, or offset coverage")
    if manifest and (shape != manifest["shape"] or repeats != manifest["repeats"]):
        raise ValueError("log shape/repeats do not match the supplied build manifest")
    if manifest and manifest["host"] != (platform == "host"):
        raise ValueError("build manifest platform does not match the requested log platform")
    samples = []
    expected = []
    for round_id in range(repeats):
        for order in range(2):
            expected.append((round_id, "optimized" if order ^ (round_id & 1) else "baseline", order))
    for line in lines:
        if not line.startswith("[dequant-ab] sample "):
            continue
        sample = fields(line)
        for key in ("round", "order", "cycles", "mismatches", "nonfinite", "guard_errors", "input_errors"):
            sample[key] = int(sample[key], 16)
        for key in ("max_abs", "mean_abs"):
            sample[key] = struct.unpack(">d", bytes.fromhex(sample.pop(key + "_f64_bits")))[0]
        if sample["status"] != "PASS" or any(sample[k] != 0 for k in
                ("mismatches", "nonfinite", "guard_errors", "input_errors", "max_abs", "mean_abs")):
            raise ValueError(f"numerical/guard/input validation failed: {sample}")
        if sample["cycles"] < 1:
            raise ValueError("nonpositive timing sample")
        samples.append(sample)
    observed = [(s["round"], s["variant"], s["order"]) for s in samples]
    if observed != expected:
        raise ValueError(f"incomplete or unordered samples: {observed!r}")
    pairs = [fields(line) for line in lines if line.startswith("[dequant-ab] pair ")]
    if [(int(p["round"], 16), int(p["mismatches"], 16)) for p in pairs] != [(i, 0) for i in range(repeats)]:
        raise ValueError("missing or failed bitwise baseline/optimized pair check")
    if lines[-1] != "[dequant-ab] PASS errors=00000000":
        raise ValueError("missing successful final completion record")
    if len(lines) != 2 + repeats * 3:
        raise ValueError("unexpected extra or duplicated A/B records")
    if platform == "fpga":
        if "[nr] RA returned: PASS" not in text or "[nr] RA returned: FAIL" in text:
            raise ValueError("missing successful NR completion")
    summary = {}
    for variant in ("baseline", "optimized"):
        times = [s["cycles"] for s in samples if s["variant"] == variant]
        summary[variant] = {"samples": times, "min": min(times),
                            "median": statistics.median(times), "mean": statistics.mean(times)}
    summary["median_speedup"] = summary["baseline"]["median"] / summary["optimized"]["median"]
    return {"status": "PASS", "platform": platform,
            "timing_unit": "RA_cycles" if platform == "fpga" else "host_clock_ticks",
            "shape": shape, "repeats": repeats, "samples": samples, "timing": summary,
            "scope": "descriptor adapter and kernel; excludes initialization, oracle, and UART",
            "correctness": "exact FP32 bits, finite results, input preservation, output guards"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--platform", choices=("fpga", "host"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = parse(args.log.read_text(errors="replace"), platform=args.platform,
                       manifest=json.loads(args.manifest.read_text()))
    except (ValueError, KeyError, IndexError, struct.error) as error:
        result = {"status": "FAIL", "reason": str(error)}
    result.update(log_sha256=hashlib.sha256(args.log.read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
