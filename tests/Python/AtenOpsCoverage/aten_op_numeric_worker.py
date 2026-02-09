#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import runpy
from pathlib import Path


def main() -> int:
    coverage_json = sys.argv[1]
    templates_source = sys.argv[2]
    strict_native = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root / "build" / "python_packages"))
    sys.path.insert(
        0,
        str(
            repo_root
            / "llvm"
            / "build"
            / "tools"
            / "mlir"
            / "python_packages"
            / "mlir_core"
        ),
    )
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import aten_op_batch_runner as runner

    templates_mod = runpy.run_path(str(Path(templates_source).resolve()))
    templates = templates_mod["CUSTOM_TEMPLATES"]

    coverage_map = runner.load_coverage_map(coverage_json)
    compiler = runner._make_compiler(strict_native=strict_native)

    for line in sys.stdin:
        name = line.strip()
        if not name:
            continue
        entry = coverage_map[name]
        result = runner.run_aten_op_numeric(
            name,
            entry,
            compiler,
            templates,
            strict_native=strict_native,
        )
        print(
            json.dumps(
                {
                    "name": result.name,
                    "status": result.status,
                    "reason": result.reason,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        compiler = runner._reset_dynamo_and_compiler(
            strict_native=strict_native
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
