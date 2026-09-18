#!/usr/bin/env python3
"""Check exact quantize A/B UART samples and archive image/source provenance."""
import re
import statistics
from archive_dequant_ab import main


def parse(text, *, platform, manifest):
    lines = [line.strip() for line in text.splitlines() if "[quant-ab]" in line]
    config = re.fullmatch(r"\[quant-ab\] rows=([0-9A-F]{8}) cols=([0-9A-F]{8})", lines[0]) if lines else None
    if not config or [int(x, 16) for x in config.groups()] != manifest["shape"]:
        raise ValueError("missing/mismatched quantize shape")
    repeats = manifest["repeats"]
    if repeats < 2 or len(lines) != 2 + 2 * repeats or manifest["host"] != (platform == "host"):
        raise ValueError("incomplete, duplicate, or wrong-platform samples")
    samples = []
    for i, line in enumerate(lines[1:-1]):
        match = re.fullmatch(r"\[quant-ab\] round=([0-9A-F]{8}) variant=(baseline|optimized) cycles=([0-9A-F]{16}) errors=([0-9A-F]{8})", line)
        if not match:
            raise ValueError("invalid quantize sample")
        round_id, variant, cycles, errors = match.groups()
        expected = "optimized" if (i % 2) ^ ((i // 2) & 1) else "baseline"
        if int(round_id, 16) != i // 2 or variant != expected or int(errors, 16) != 0 or int(cycles, 16) <= 0:
            raise ValueError("failed, unordered, or invalid timing sample")
        samples.append({"round": i // 2, "variant": variant, "cycles": int(cycles, 16), "errors": 0})
    if lines[-1] != "[quant-ab] PASS errors=00000000" or "FAIL" in text:
        raise ValueError("missing successful exact A/B and oracle completion")
    if platform == "fpga" and (text.count("[nr] RA returned: PASS") != 1 or text.count("verify NR runtime: PASS") != 1):
        raise ValueError("missing/duplicate NR completion")
    timing = {}
    for variant in ("baseline", "optimized"):
        values = [s["cycles"] for s in samples if s["variant"] == variant]
        timing[variant] = {"samples": values, "min": min(values),
                           "mean": statistics.mean(values), "median": statistics.median(values)}
    timing["mean_speedup"] = timing["baseline"]["mean"] / timing["optimized"]["mean"]
    return {"status": "PASS", "shape": manifest["shape"], "samples": samples, "timing": timing,
            "platform": platform, "repeats": repeats,
            "timing_unit": "RA_cycles" if platform == "fpga" else "host_clock_ticks",
            "scope": "exact int8, scale bits, nonzero descriptor offsets, guards and input integrity; kernel-only A/B"}


if __name__ == "__main__":
    main(benchmark="quant", parse_log=parse)
