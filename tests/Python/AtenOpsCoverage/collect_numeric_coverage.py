#!/usr/bin/env python3
# RUN: %PYTHON %s --help
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import runpy
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CATALOG = THIS_DIR / "aten_op_catalog.json"
DEFAULT_UNIQUE_OPS = THIS_DIR / "aten_op_unique_ops.json"
DEFAULT_OUT_BY_OP = THIS_DIR / "aten_op_numeric_coverage_by_op.json"
DEFAULT_OUT_BY_OVERLOAD = THIS_DIR / "aten_op_numeric_results_by_overload.json"


def _bootstrap_pythonpath() -> None:
    os.environ.setdefault("BUDDY_OC_VALIDATE_NUMERIC", "1")
    os.environ.setdefault("BUDDY_RNG_SEED", "0")

    sys.path.insert(0, str(REPO_ROOT / "build" / "python_packages"))
    sys.path.insert(
        0,
        str(
            REPO_ROOT
            / "llvm"
            / "build"
            / "tools"
            / "mlir"
            / "python_packages"
            / "mlir_core"
        ),
    )
    sys.path.insert(0, str(THIS_DIR))


def _load_batch_vars(path: Path) -> Tuple[List[str], Dict[str, Any]]:
    mod = runpy.run_path(str(path), run_name="__batch__")
    ops = list(mod.get("OPS", []))
    templates = dict(mod.get("CUSTOM_TEMPLATES", {}) or {})
    return ops, templates


def _pick_best_overload(overloads: Iterable[str]) -> str:
    priority = (
        "default",
        "Tensor",
        "self",
        "Tensor_out",
        "out",
    )
    overloads = list(overloads)
    for want in priority:
        if want in overloads:
            return want
    return overloads[0] if overloads else ""


@dataclass(frozen=True)
class OverloadResult:
    name: str  # op.overload
    op: str
    overload: str
    status: str  # pass | skip | fail
    reason: str = ""


@dataclass(frozen=True)
class OpAggregate:
    op: str
    status: str  # pass | skip | fail
    pass_overload_count: int
    fail_overload_count: int
    skip_overload_count: int
    sample_pass_overload: str = ""
    sample_fail_overload: str = ""
    sample_fail_reason: str = ""


def _run_all_batches(
    batch_files: List[Path],
    coverage_json: Path,
) -> List[OverloadResult]:
    import aten_op_batch_runner as runner

    all_results: List[OverloadResult] = []
    silent_out = io.StringIO()

    for batch_path in batch_files:
        ops, templates = _load_batch_vars(batch_path)
        with contextlib.redirect_stdout(silent_out), contextlib.redirect_stderr(
            silent_out
        ):
            results = runner.run_aten_op_batch(
                ops,
                coverage_json=coverage_json,
                batch_label=batch_path.stem,
                max_fails=0,
                templates=templates,
                show_skips=True,
                validate_numeric=True,
                templates_source=batch_path,
            )
        for r in results:
            op, overload = r.name.split(".", 1)
            all_results.append(
                OverloadResult(
                    name=r.name,
                    op=op,
                    overload=overload,
                    status=r.status,
                    reason=r.reason or "",
                )
            )

    return all_results


def _aggregate_by_op(
    overload_results: List[OverloadResult],
) -> List[OpAggregate]:
    by_op: Dict[str, List[OverloadResult]] = defaultdict(list)
    for r in overload_results:
        by_op[r.op].append(r)

    out: List[OpAggregate] = []
    for op, rows in sorted(by_op.items(), key=lambda kv: kv[0]):
        passed = [r for r in rows if r.status == "pass"]
        failed = [r for r in rows if r.status == "fail"]
        skipped = [r for r in rows if r.status == "skip"]

        if passed:
            status = "pass"
        elif failed:
            status = "fail"
        else:
            status = "skip"

        sample_pass = (
            _pick_best_overload([r.overload for r in passed]) if passed else ""
        )
        sample_fail_overload = failed[0].overload if failed else ""
        sample_fail_reason = failed[0].reason if failed else ""
        out.append(
            OpAggregate(
                op=op,
                status=status,
                pass_overload_count=len(passed),
                fail_overload_count=len(failed),
                skip_overload_count=len(skipped),
                sample_pass_overload=sample_pass,
                sample_fail_overload=sample_fail_overload,
                sample_fail_reason=sample_fail_reason,
            )
        )

    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_unique_ops(path: Path) -> List[str]:
    items = json.loads(path.read_text("utf-8"))
    return [it["op"] for it in items]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect AtenOpsCoverage numeric validation results and aggregate by unique op."
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help="Path to aten_op_catalog.json",
    )
    parser.add_argument(
        "--unique-ops",
        type=Path,
        default=DEFAULT_UNIQUE_OPS,
        help="Path to aten_op_unique_ops.json (701 op baseline)",
    )
    parser.add_argument(
        "--out-by-op",
        type=Path,
        default=DEFAULT_OUT_BY_OP,
        help="Output JSON aggregated by unique op",
    )
    parser.add_argument(
        "--out-by-overload",
        type=Path,
        default=DEFAULT_OUT_BY_OVERLOAD,
        help="Output JSON with per-overload results",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _bootstrap_pythonpath()

    batch_files = sorted(THIS_DIR.glob("test_aten_op_batch_*.py"))
    overload_results = _run_all_batches(batch_files, args.catalog)

    # Basic validation: one result per catalog entry (op.overload).
    seen = {r.name for r in overload_results}
    if len(seen) != len(overload_results):
        raise RuntimeError("duplicate_results_detected")

    catalog_entries = json.loads(args.catalog.read_text("utf-8"))
    expected_names = {f"{e['op']}.{e['overload']}" for e in catalog_entries}
    missing = sorted(expected_names - seen)
    extra = sorted(seen - expected_names)
    if missing or extra:
        raise RuntimeError(
            f"result_name_mismatch missing={len(missing)} extra={len(extra)}"
        )

    agg = _aggregate_by_op(overload_results)

    unique_ops = _load_unique_ops(args.unique_ops)
    agg_ops = [a.op for a in agg]
    if set(unique_ops) != set(agg_ops):
        raise RuntimeError(
            f"unique_op_mismatch unique_ops={len(unique_ops)} agg_ops={len(agg_ops)}"
        )

    # Write results.
    _write_json(
        args.out_by_overload,
        [
            {
                "name": r.name,
                "op": r.op,
                "overload": r.overload,
                "status": r.status,
                "reason": r.reason,
            }
            for r in sorted(overload_results, key=lambda r: r.name)
        ],
    )
    _write_json(
        args.out_by_op,
        [
            {
                "op": a.op,
                "status": a.status,
                "pass_overload_count": a.pass_overload_count,
                "fail_overload_count": a.fail_overload_count,
                "skip_overload_count": a.skip_overload_count,
                "sample_pass_overload": a.sample_pass_overload,
                "sample_fail_overload": a.sample_fail_overload,
                "sample_fail_reason": a.sample_fail_reason,
            }
            for a in agg
        ],
    )

    # Print summary.
    passed = sum(1 for a in agg if a.status == "pass")
    failed = sum(1 for a in agg if a.status == "fail")
    skipped = sum(1 for a in agg if a.status == "skip")
    total = len(agg)
    rate = (passed / total) if total else 0.0
    print(
        f"UNIQUE_OP_SUMMARY pass={passed} fail={failed} skip={skipped} total={total} pass_rate={rate:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
