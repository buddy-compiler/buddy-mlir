#!/usr/bin/env python3
# ===- run_coverage.py - PyTorch op coverage CLI for Buddy-MLIR ------------===
#
# Licensed under the Apache License, Version 2.0 (the "License").
# ===----------------------------------------------------------------------===
"""
Automated PyTorch operator coverage analysis for Buddy-MLIR (issue #911).

Examples:
  # Static analysis (no torch / buddy build required)
  python tools/pytorch_op_coverage/run_coverage.py

  # Custom output dir
  python tools/pytorch_op_coverage/run_coverage.py --out-dir /tmp/coverage-out

  # Live probes when PYTHONPATH includes buddy + mlir bindings
  python tools/pytorch_op_coverage/run_coverage.py --mode live
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Allow running as a script without installing a package.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from classify import OpCoverageRecord, classify_static  # noqa: E402
from parse_frontend import (  # noqa: E402
    default_frontend_py,
    default_ops_dir,
    flatten_target_ops,
    load_target_op_set,
    merge_registry_keys,
    parse_all_registries,
    parse_ops_map,
)
from report import build_report_payload, write_json, write_markdown  # noqa: E402


def _repo_root() -> Path:
    return _HERE.parents[1]


def _git_rev(repo: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def run_static(repo: Path, target_path: Path) -> list[OpCoverageRecord]:
    ops_map = parse_ops_map(default_frontend_py(repo))
    dialects = parse_all_registries(default_ops_dir(repo))
    lowering = merge_registry_keys(dialects)
    target = load_target_op_set(target_path)
    known_partial = target.get("known_partial_or_limited", {})
    decomp_aliases = target.get("decomp_aliases", {})
    rows = flatten_target_ops(target)
    return [
        classify_static(
            aten=row["aten"],
            family=row["family"],
            families=row.get("families"),
            ops_map=ops_map,
            lowering_by_op=lowering,
            known_partial=known_partial,
            decomp_aliases=decomp_aliases,
        )
        for row in rows
    ]


def run_live(records: list[OpCoverageRecord]) -> tuple[list[OpCoverageRecord], dict]:
    from probes import _try_import_stack, run_live_probe, seed_probes

    ok, msg = _try_import_stack()
    extra = {"live_stack": msg}
    if not ok:
        for r in records:
            r.live_error = msg
            if r.status in ("fully_supported_static", "partial", "frontend_only"):
                # Keep static status; annotate skip.
                r.notes = (r.notes + f" | live skipped: {msg}").strip(" |")
        extra["live_ran"] = False
        return records, extra

    probes = seed_probes()
    extra["live_ran"] = True
    extra["probed"] = []
    for r in records:
        probe = probes.get(r.aten)
        if probe is None:
            continue
        result = run_live_probe(r.aten, probe)
        r.lowered = result.get("lowered", "not_run")
        r.compiled = result.get("compiled", "not_run")
        r.correctness = result.get("correctness", "not_run")
        r.live_error = result.get("error")
        if result.get("error"):
            r.status = "live_failed"
            r.notes = (r.notes + f" | live error: {result['error']}").strip(" |")
        elif result.get("lowered") == "yes":
            r.status = "live_passed"
            r.notes = (r.notes + " | live lower ok").strip(" |")
        extra["probed"].append({"aten": r.aten, **result})
    return records, extra


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Buddy-MLIR PyTorch operator coverage tool (issue #911)"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_repo_root(),
        help="buddy-mlir repository root",
    )
    parser.add_argument(
        "--target-set",
        type=Path,
        default=_HERE / "data" / "target_ops_v0.json",
        help="JSON defining the coverage denominator",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_HERE / "out",
        help="Directory for JSON + Markdown reports",
    )
    parser.add_argument(
        "--mode",
        choices=("static", "live"),
        default="static",
        help="static=parse frontend sources; live=also run DynamoCompiler probes",
    )
    args = parser.parse_args(argv)

    repo = args.repo_root.resolve()
    target_path = args.target_set.resolve()
    out_dir = args.out_dir.resolve()

    records = run_static(repo, target_path)
    extra = {
        "frontend_ops_map_size": len(parse_ops_map(default_frontend_py(repo))),
        "registry_op_classes": len(
            merge_registry_keys(parse_all_registries(default_ops_dir(repo)))
        ),
    }
    if args.mode == "live":
        records, live_extra = run_live(records)
        extra.update(live_extra)

    payload = build_report_payload(
        records,
        mode=args.mode,
        repo_rev=_git_rev(repo),
        target_set_path=str(target_path),
        extra=extra,
    )
    json_path = out_dir / "pytorch_op_coverage.json"
    md_path = out_dir / "pytorch_op_coverage.md"
    write_json(payload, json_path)
    write_markdown(payload, md_path)

    s = payload["summary"]
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        f"Target ops={s['total_ops']}  "
        f"frontend={s['frontend_recognized']} ({s['frontend_recognized_pct']}%)  "
        f"static_full={s['fully_supported_static']} ({s['fully_supported_static_pct']}%)  "
        f"unsupported={s['unsupported']}"
    )
    print(s["disclaimer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
