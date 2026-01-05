#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_module_from_file(path: Path):
    spec = importlib.util.spec_from_file_location("buddy_oc_templates", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load templates from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-json", required=True)
    parser.add_argument("--templates-source", required=True)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))

    import aten_op_batch_runner as runner

    templates_path = Path(args.templates_source).resolve()
    templates_mod = _load_module_from_file(templates_path)
    templates = getattr(templates_mod, "CUSTOM_TEMPLATES", {})
    if not isinstance(templates, dict):
        raise RuntimeError(
            f"CUSTOM_TEMPLATES must be a dict in {templates_path}"
        )

    coverage_map = runner.load_coverage_map(args.coverage_json)
    compiler = runner._make_compiler()

    for line in sys.stdin:
        name = line.strip()
        if not name:
            continue
        entry = coverage_map.get(
            name, {"op": name, "overload": "", "notes": "missing_in_coverage"}
        )
        result = runner.run_aten_op_numeric(name, entry, compiler, templates)
        payload = {
            "name": result.name,
            "status": result.status,
            "reason": result.reason,
        }
        print(runner._WORKER_RESULT_PREFIX + json.dumps(payload), flush=True)
        compiler = runner._reset_dynamo_and_compiler()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
