#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CATALOG = THIS_DIR / "aten_op_catalog.json"
DEFAULT_UNIQUE_OPS = THIS_DIR / "aten_op_unique_ops.json"
DEFAULT_OUT = THIS_DIR / "aten_op_unique_native_numeric_results.json"
DEFAULT_TEMPLATES_SOURCE = THIS_DIR / "aten_op_unique_native_templates.py"

# Unique-op representative overloads: prefer Dynamo-capturable tensor forms
# when op.default is scalar/list-return and does not represent real lowering.
PREFERRED_UNIQUE_OVERLOAD: Dict[str, str] = {
    "eq": "Tensor",
    "ge": "Tensor",
    "gt": "Tensor",
    "le": "Tensor",
    "lt": "Tensor",
    "ne": "Tensor",
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


def _pick_best_overload(op: str, overloads: Iterable[str]) -> str:
    priority = (
        "default",
        "Tensor",
        "self",
        "Tensor_out",
        "out",
    )
    items = list(overloads)
    preferred = PREFERRED_UNIQUE_OVERLOAD.get(op)
    if preferred and preferred in items:
        return preferred
    for want in priority:
        if want in items:
            return want
    return items[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run numeric validation for unique ATen ops in strict native mode (no shim)."
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
        help="Path to aten_op_unique_ops.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Path to write overload-level strict-native results",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _bootstrap_pythonpath()

    import aten_op_batch_runner as runner

    catalog_entries = json.loads(args.catalog.read_text("utf-8"))
    unique_items = json.loads(args.unique_ops.read_text("utf-8"))
    coverage_map = runner.load_coverage_map(args.catalog)

    by_op: Dict[str, List[str]] = {}
    for entry in catalog_entries:
        by_op.setdefault(entry["op"], []).append(entry["overload"])

    selected_names: List[str] = []
    for item in unique_items:
        op = item["op"]
        if op == "normal" and "Tensor_float" in by_op[op]:
            overload = "Tensor_float"
        else:
            overload = _pick_best_overload(op, by_op[op])
        selected_names.append(f"{op}.{overload}")

    results = runner.run_aten_op_batch(
        selected_names,
        coverage_json=args.catalog,
        batch_label="unique_native_numeric",
        max_fails=0,
        templates={},
        show_skips=False,
        validate_numeric=True,
        templates_source=DEFAULT_TEMPLATES_SOURCE,
        numeric_strict_native=True,
    )

    rows: List[Dict[str, Any]] = []
    pass_count = 0
    fail_count = 0
    skip_count = 0

    for result in results:
        entry = coverage_map[result.name]
        rows.append(
            {
                "name": result.name,
                "op": entry["op"],
                "overload": entry["overload"],
                "status": result.status,
                "reason": result.reason,
            }
        )

        if result.status == "pass":
            pass_count += 1
        elif result.status == "fail":
            fail_count += 1
        else:
            skip_count += 1

    args.out.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    total = len(rows)
    pass_rate = (pass_count / total) if total else 0.0
    print(
        f"UNIQUE_NATIVE_SUMMARY pass={pass_count} fail={fail_count} "
        f"skip={skip_count} total={total} pass_rate={pass_rate:.4f}"
    )
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
