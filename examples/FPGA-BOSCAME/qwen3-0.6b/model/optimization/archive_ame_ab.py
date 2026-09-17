#!/usr/bin/env python3
"""Archive compact evidence from a successful fixed/direct AME board A/B run."""
import argparse
import json
from pathlib import Path
import re
import shutil
import statistics
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dequant_ab import inspect_case, sha


def parse(log, manifest, platform="fpga"):
    lines = [line.strip() for line in log.splitlines() if "[ame-ab]" in line]
    cfg = [line for line in lines if line.startswith("[ame-ab] config ")]
    if len(cfg) != 1 or not lines or lines[-1] != "[ame-ab] PASS errors=00000000":
        raise ValueError("missing AME configuration or final PASS")
    fields = lambda line: dict(re.findall(r"(\w+)=(\S+)", line))
    c = fields(cfg[0])
    shape = [int(c[k], 16) for k in ("m", "n", "k")]
    if shape != manifest["shape"] or int(c["repeats"], 16) != manifest["repeats"]:
        raise ValueError("shape/repeat mismatch")
    if c.get("semantic") != "accumulate" or c.get("descriptor_offset_bytes") != "00000040":
        raise ValueError("semantic/descriptor coverage missing")
    if c.get("input_check") != manifest["input_integrity"]:
        raise ValueError("input integrity mode mismatch")
    if manifest["host"] != (platform == "host"):
        raise ValueError("manifest/log platform mismatch")
    if platform == "fpga" and ("[nr] RA returned: PASS" not in log or "[nr] RA returned: FAIL" in log):
        raise ValueError("missing successful NR return")
    samples = [fields(line) for line in lines if line.startswith("[ame-ab] sample ")]
    expected = [(r, "optimized" if o ^ (r & 1) else "baseline", o)
                for r in range(manifest["repeats"]) for o in range(2)]
    actual = [(int(s["round"], 16), s["variant"], int(s["order"], 16)) for s in samples]
    if actual != expected or len(lines) != 2 + manifest["repeats"] * 4:
        raise ValueError("missing, duplicated, or unordered AME samples")
    for s in samples:
        for key in ("round", "order", "cycles", "adapter_kernel_cycles", "pre_sync_cycles", "post_sync_cycles",
                    "mismatches", "guard_errors", "input_errors", "max_abs_i64"):
            s[key] = int(s[key], 16)
        s["mean_abs"] = struct.unpack(">d", bytes.fromhex(s.pop("mean_abs_f64_bits")))[0]
        for key in ("cycles", "adapter_kernel_cycles", "pre_sync_cycles", "post_sync_cycles"):
            if s[key] < 1 and key == "adapter_kernel_cycles":
                raise ValueError("nonpositive kernel timing")
        if any(s[k] != 0 for k in ("mismatches", "guard_errors", "input_errors", "max_abs_i64", "mean_abs")) or s["status"] != "PASS":
            raise ValueError("AME numerical/input/guard failure")
        if s["cycles"] != s["adapter_kernel_cycles"] + s["post_sync_cycles"]:
            raise ValueError("total timing does not equal kernel plus post sync")
    preparation = [fields(line) for line in lines if line.startswith("[ame-ab] prepare ")]
    if [int(s["round"], 16) for s in preparation] != list(range(manifest["repeats"])):
        raise ValueError("missing changed-input round preparation")
    pairs = [fields(line) for line in lines if line.startswith("[ame-ab] pair ")]
    if [(int(p["round"], 16), int(p["mismatches"], 16)) for p in pairs] != [(r, 0) for r in range(manifest["repeats"])]:
        raise ValueError("pair check failed")
    timing = {}
    for variant in ("baseline", "optimized"):
        values = [s["cycles"] for s in samples if s["variant"] == variant]
        timing[variant] = {"cycles": values, "median": statistics.median(values), "mean": statistics.mean(values)}
    timing["median_speedup"] = timing["baseline"]["median"] / timing["optimized"]["median"]
    return {"status": "PASS", "shape": shape, "repeats": manifest["repeats"], "samples": samples,
            "platform": platform, "timing_unit": "RA_cycles" if platform == "fpga" else "host_clock_ticks",
            "timing": timing, "scope": "adapter+kernel+post ame_fence; pre-sync separately recorded",
            "correctness": "exact int64 oracle, all outputs, guards",
            "input_integrity": c["input_check"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path); p.add_argument("--build", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path); args = p.parse_args()
    manifest_path = args.build / "manifest.json"; manifest = json.loads(manifest_path.read_text())
    runner = json.loads((args.run / "result.json").read_text())
    image = args.build / manifest["image"]["path"]
    if sha(image) != manifest["image"]["sha256"] or sha(image) != runner["sha256"]:
        raise ValueError("image hash mismatch")
    if runner["status"] != "OK" or runner.get("ddr_readback_matches") is not True or not runner.get("completion_marker_seen"):
        raise ValueError("upload/readback/completion failed")
    if (args.run / "uart.raw.log").stat().st_size != runner.get("uart_bytes"):
        raise ValueError("UART byte count differs from runner result")
    result = parse((args.run / "uart.raw.log").read_text(errors="replace"), manifest)
    native_manifests = {}
    result["variant_configuration"] = {}
    for variant, entry in manifest["variants"].items():
        source = Path(entry["source"]); inspect_case(source, "ame")
        for relative, expected in entry["inputs_sha256"].items():
            if sha(source / relative) != expected or sha(args.build / "inputs" / variant / relative) != expected:
                raise ValueError(f"source/snapshot changed: {variant}/{relative}")
        native = args.build / "inputs" / variant / "nr/manifest.json"
        native_manifests[variant] = native
        native_data = json.loads(native.read_text())
        variables = native_data["configuration"]["variables"]
        result["variant_configuration"][variant] = {
            "ame_gpr_mode": native_data.get("ame_gpr_mode", variables.get("AME_GPR_MODE", "fixed")),
            "coalesce_fences": native_data.get("nr_coalesce_fences", variables.get("NR_COALESCE_FENCES", "0")),
            "tile": {key: entry["manifest"]["constexprs"][key] for key in ("BM", "BN", "BK")},
            "grid": entry["manifest"]["grid"],
            "native_manifest_sha256": sha(native)}
    if args.output.exists() and any(args.output.iterdir()): raise ValueError("nonempty output archive")
    args.output.mkdir(parents=True, exist_ok=True)
    for source, name in ((args.run/"uart.raw.log", "uart.raw.log"), (args.run/"result.json", "runner-result.json"),
                         (manifest_path, "build-manifest.json"), (args.build/"elf-audit.json", "elf-audit.json")):
        shutil.copy2(source, args.output/name)
    for variant, source in native_manifests.items():
        shutil.copy2(source, args.output/f"{variant}-native-manifest.json")
    result.update(run_id=args.run.name, image_sha256=sha(image), log_sha256=sha(args.run/"uart.raw.log"),
                  manifest_sha256=sha(manifest_path), ddr_readback_matches=True,
                  source_hashes_and_abi_checked=True)
    (args.output/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
