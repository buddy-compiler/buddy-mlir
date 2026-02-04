#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import runpy
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEST_DIR = Path(__file__).resolve().parent


def _bootstrap_pythonpath() -> None:
    os.environ["BUDDY_OC_VALIDATE_NUMERIC"] = "1"
    os.environ["BUDDY_RNG_SEED"] = "0"

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
    sys.path.insert(0, str(DEFAULT_TEST_DIR))


def _load_batch_vars(path: Path):
    # Avoid executing the file's __main__ block (which would run in non-numeric mode).
    mod = runpy.run_path(str(path), run_name="__batch__")
    return mod["OPS"], mod.get("CUSTOM_TEMPLATES", {})


def _run_one_batch(path: Path) -> int:
    _bootstrap_pythonpath()

    import aten_op_batch_runner as runner

    ops, templates = _load_batch_vars(path)
    results = runner.run_aten_op_batch(
        ops,
        batch_label=path.stem,
        max_fails=20,
        templates=templates,
        templates_source=path,
        show_skips=True,
        validate_numeric=True,
    )
    return 1 if any(r.status == "fail" for r in results) else 0


def main() -> int:
    batch_dir = DEFAULT_TEST_DIR
    batch_files = sorted(batch_dir.glob("test_aten_op_batch_*.py"))

    failed = 0
    for path in batch_files:
        if _run_one_batch(path) != 0:
            failed += 1
            print(f"BATCH_FAIL {path.name}")

    print(f"BATCH_SUMMARY fail={failed} count={len(batch_files)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
