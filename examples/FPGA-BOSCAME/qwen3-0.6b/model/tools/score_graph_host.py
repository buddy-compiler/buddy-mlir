#!/usr/bin/env python3
"""Score captured graph output against a chosen independent quantized oracle."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_graph_host import compare_reference


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--quant-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-abs-error", type=float, default=1e-3)
    parser.add_argument("--mean-abs-error", type=float, default=1e-4)
    args = parser.parse_args()
    run_path = args.run / "host-run.json"
    reference_path = args.quant_reference / "quant-reference.json"
    run = json.loads(run_path.read_text())
    reference = json.loads(reference_path.read_text())
    for key in ("layers", "prompt_ids"):
        if run[key] != reference[key]:
            raise SystemExit(f"different {key}: refusing an unrelated comparison")
    if run["max_cache_len"] != reference["capacity"]:
        raise SystemExit("different cache capacity")
    if len(run["decode_steps_recorded"]) != len(reference["decode_steps_recorded"]):
        raise SystemExit("different decode-step counts")
    arrays = np.load(args.run / "arrays.npz")
    result = compare_reference(arrays, run, args.quant_reference, quantized=True)
    required_caches = 2 + 2 * len(run["decode_steps_recorded"])
    complete_cache = len(result["cache_snapshots_compared"]) == required_caches
    passed = (result["token_trajectory_match"] and all(result["decode_same_input_context"])
              and complete_cache
              and all("max_abs_error" in value
                      and value["max_abs_error"] <= args.max_abs_error
                      and value["mean_abs_error"] <= args.mean_abs_error
                      for value in result["comparisons"].values()))
    result.update({"status": "PASS" if passed else "FAIL", "not_fpga": True,
                   "scope": "captured Buddy compiled graph execution versus independent host quantized reference",
                   "layers": run["layers"], "prompt_ids": run["prompt_ids"],
                   "cache_snapshots_complete": complete_cache,
                   "max_abs_error_limit": args.max_abs_error,
                   "mean_abs_error_limit": args.mean_abs_error,
                   "sources": [fingerprint(path) for path in
                               (run_path, args.run / "arrays.npz", reference_path,
                                args.quant_reference / "arrays.npz")]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['status']}: {run['layers']} layers, {len(result['comparisons'])} tensor comparisons")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
