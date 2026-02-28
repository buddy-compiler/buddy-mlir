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

DEFAULT_CATALOG = THIS_DIR / "aten_coverage_catalog.json"
DEFAULT_UNIQUE_OPS = THIS_DIR / "aten_coverage_unique_ops.json"
DEFAULT_OUT_BY_OP = THIS_DIR / "aten_coverage_by_op.json"
DEFAULT_OUT_BY_OVERLOAD = THIS_DIR / "aten_coverage_by_overload.json"


NUMERIC_UNIQUE_OVERLOAD_OVERRIDES = {
    "add": "Tensor",
    "copysign": "Tensor",
    "div": "Tensor",
    "eq": "Tensor",
    "fmod": "Tensor",
    "frexp": "Tensor",
    "ge": "Tensor",
    "gt": "Tensor",
    "le": "Tensor",
    "lt": "Tensor",
    "mul": "Tensor",
    "ne": "Tensor",
    "normal": "Tensor_Tensor",
    "norm": "ScalarOpt_dim",
    "rand": "default",
    "randn": "default",
    "remainder": "Tensor",
    "repeat_interleave": "Tensor",
    "select": "int",
    "sub": "Tensor",
    "where": "self",
}


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


def _load_unique_op_names(path: Path) -> List[str]:
    items = json.loads(path.read_text("utf-8"))
    return [it["op"] for it in items]


def _pick_ops_for_unique(
    catalog_entries: List[Dict[str, Any]],
    unique_ops: List[str],
    mode: str,
) -> List[str]:
    by_op: Dict[str, List[str]] = defaultdict(list)
    for entry in catalog_entries:
        by_op[entry["op"]].append(entry["overload"])

    selected: List[str] = []
    for op in unique_ops:
        overloads = by_op.get(op, [])
        if not overloads:
            continue
        if mode == "numeric":
            forced = NUMERIC_UNIQUE_OVERLOAD_OVERRIDES.get(op)
            if forced and forced in overloads:
                selected.append(f"{op}.{forced}")
                continue
        overload = _pick_best_overload(overloads)
        selected.append(f"{op}.{overload}")
    return selected


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
    mode: str,
) -> List[OverloadResult]:
    import aten_coverage_runner as runner

    all_results: List[OverloadResult] = []
    silent_out = io.StringIO()

    for batch_path in batch_files:
        ops, templates = _load_batch_vars(batch_path)
        with contextlib.redirect_stdout(silent_out), contextlib.redirect_stderr(
            silent_out
        ):
            results = runner.run_aten_coverage_batch(
                ops,
                coverage_json=coverage_json,
                batch_label=batch_path.stem,
                max_fails=0,
                templates=templates,
                show_skips=True,
                mode=mode,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AtenOpsCoverage in graph/numeric mode and aggregate results."
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help="Path to aten_coverage_catalog.json",
    )
    parser.add_argument(
        "--unique-ops",
        type=Path,
        default=DEFAULT_UNIQUE_OPS,
        help="Path to aten_coverage_unique_ops.json",
    )
    parser.add_argument(
        "--mode",
        choices=("numeric", "graph"),
        default="numeric",
        help="Validation mode",
    )
    parser.add_argument(
        "--scope",
        choices=("all", "unique", "ops"),
        default="unique",
        help="Operator set scope",
    )
    parser.add_argument(
        "--ops",
        type=str,
        default="",
        help="Comma-separated op.overload names for --scope ops",
    )
    parser.add_argument(
        "--emit-json",
        action="store_true",
        help="Write aggregated JSON outputs to default paths",
    )
    parser.add_argument(
        "--out-by-op",
        type=Path,
        default=None,
        help="Optional output JSON path aggregated by unique op",
    )
    parser.add_argument(
        "--out-by-overload",
        type=Path,
        default=None,
        help="Optional output JSON path with per-overload results",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _bootstrap_pythonpath()

    if args.scope == "ops":
        op_names = [x.strip() for x in args.ops.split(",") if x.strip()]
        if not op_names:
            raise RuntimeError("scope_ops_requires_non_empty_--ops")

        import aten_coverage_runner as runner

        results = runner.run_aten_coverage_batch(
            op_names,
            coverage_json=args.catalog,
            batch_label="ops_scope",
            max_fails=0,
            templates={},
            show_skips=True,
            mode=args.mode,
        )
        overload_results = [
            OverloadResult(
                name=r.name,
                op=r.name.split(".", 1)[0],
                overload=r.name.split(".", 1)[1],
                status=r.status,
                reason=r.reason or "",
            )
            for r in results
        ]
    elif args.scope == "unique":
        catalog_entries = json.loads(args.catalog.read_text("utf-8"))
        unique_ops = _load_unique_op_names(args.unique_ops)
        op_names = _pick_ops_for_unique(
            catalog_entries,
            unique_ops,
            args.mode,
        )

        import aten_coverage_runner as runner

        results = runner.run_aten_coverage_batch(
            op_names,
            coverage_json=args.catalog,
            batch_label="unique_scope",
            max_fails=0,
            templates={},
            show_skips=True,
            mode=args.mode,
        )
        overload_results = [
            OverloadResult(
                name=r.name,
                op=r.name.split(".", 1)[0],
                overload=r.name.split(".", 1)[1],
                status=r.status,
                reason=r.reason or "",
            )
            for r in results
        ]
    else:
        batch_files = sorted(THIS_DIR.glob("test_aten_coverage_batch_*.py"))
        overload_results = _run_all_batches(
            batch_files, args.catalog, args.mode
        )

    seen = {r.name for r in overload_results}
    if len(seen) != len(overload_results):
        raise RuntimeError("duplicate_results_detected")

    agg = _aggregate_by_op(overload_results)

    if args.scope == "all":
        catalog_entries = json.loads(args.catalog.read_text("utf-8"))
        expected_names = {f"{e['op']}.{e['overload']}" for e in catalog_entries}
        missing = sorted(expected_names - seen)
        extra = sorted(seen - expected_names)
        if missing or extra:
            raise RuntimeError(
                f"result_name_mismatch missing={len(missing)} extra={len(extra)}"
            )

    out_by_op = args.out_by_op
    out_by_overload = args.out_by_overload
    if args.emit_json:
        if out_by_op is None:
            out_by_op = DEFAULT_OUT_BY_OP
        if out_by_overload is None:
            out_by_overload = DEFAULT_OUT_BY_OVERLOAD

    if out_by_overload is not None:
        _write_json(
            out_by_overload,
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
    if out_by_op is not None:
        _write_json(
            out_by_op,
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
