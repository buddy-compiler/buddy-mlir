#!/usr/bin/env python3
"""Worker for a single op. Prints one JSON line to stdout and exits.

Invoked by run_unique_numeric_isolated.py as a subprocess so that any
SIGSEGV / SIGABRT in LLVM JIT code only kills this worker, not the
coverage run as a whole.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]


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


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: aten_coverage_worker.py <op.overload>", file=sys.stderr)
        return 2
    name = sys.argv[1]
    _bootstrap_pythonpath()

    import aten_coverage_runner as R

    cov_map = R.load_coverage_map(THIS_DIR / "aten_coverage_catalog.json")
    entry = cov_map.get(name) or {
        "op": name.split(".", 1)[0],
        "overload": name.split(".", 1)[1],
        "notes": "missing_in_coverage",
    }
    compiler = R._make_compiler()
    try:
        res = R.run_aten_coverage_numeric(name, entry, compiler, {})
        rec = {"name": res.name, "status": res.status, "reason": res.reason}
    except SystemExit:
        raise
    except BaseException as e:
        traceback.print_exc(file=sys.stderr)
        rec = {
            "name": name,
            "status": "fail",
            "reason": f"worker_exc:{type(e).__name__}:{e}",
        }
    print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
