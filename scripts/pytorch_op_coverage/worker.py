"""Isolated export/JIT worker. Write progress before entering native stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from classify import STAGES
from probes import WORKLOADS, build_case, resolve_operator


def write_result(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def environment(operators):
    result = {
        "python": sys.version.split()[0],
        "torch": None,
        "buddy": False,
        "schemas": {},
        "schema_errors": {},
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
    }
    try:
        import torch

        result["torch"] = torch.__version__
    except Exception as exc:
        result["error"] = f"torch: {type(exc).__name__}: {exc}"
        return result
    for name in operators:
        try:
            result["schemas"][name] = str(resolve_operator(name)._schema)
        except Exception as exc:
            result["schema_errors"][name] = str(exc)
    try:
        from buddy.compiler import frontend

        result["buddy"] = True
        result["buddy_frontend_sha256"] = hashlib.sha256(
            Path(frontend.__file__).read_bytes()
        ).hexdigest()
        source_root = Path(frontend.__file__).parent
        result["buddy_source_sha256"] = {
            p.relative_to(source_root).as_posix(): hashlib.sha256(
                p.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for p in sorted(source_root.rglob("*.py"))
        }
        try:
            result["buddy_distribution_version"] = version("buddy")
        except PackageNotFoundError:
            result["buddy_distribution_version"] = None
    except Exception as exc:
        result["error"] = f"buddy.compiler: {type(exc).__name__}: {exc}"
    return result


def compare_outputs(expected, actual, torch):
    # Reuse the existing runner's tensor/scalar flattening, but require exact
    # arity and ordered numerical equality (no metadata-only or truncation path).
    from aten_coverage_runner import _flatten_outputs

    ok, error, expected_items = _flatten_outputs(expected)
    if not ok:
        raise AssertionError(error)
    ok, error, actual_items = _flatten_outputs(actual)
    if not ok:
        raise AssertionError(error)
    if len(expected_items) != len(actual_items) or not expected_items:
        raise AssertionError("Output arity mismatch or empty output")
    for (expected_kind, ref), (actual_kind, out) in zip(
        expected_items, actual_items
    ):
        if expected_kind != actual_kind:
            raise AssertionError("Output kind mismatch")
        floating = (
            ref.is_floating_point() or ref.is_complex()
            if isinstance(ref, torch.Tensor)
            else isinstance(ref, (float, complex))
        )
        torch.testing.assert_close(
            out,
            ref,
            rtol=1e-4 if floating else 0,
            atol=1e-5 if floating else 0,
            equal_nan=False,
        )


def run_case(name, profile, mode, path):
    result = {
        "case_id": profile,
        "status": "failed",
        "reason": "",
        **dict.fromkeys(STAGES, "not_run"),
        "active_stage": "setup",
    }
    write_result(path, result)
    try:
        import torch

        torch.set_num_threads(1)
        torch.manual_seed(0)
        module, args = build_case(name, profile)
        result["inputs"] = [
            {
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "stride": list(a.stride()),
            }
            for a in args
        ]
        result["active_stage"] = "exported"
        write_result(path, result)
        with torch.no_grad():
            expected = module(*args)
            exported = torch.export.export(module, args, strict=True)
            from torch.utils._pytree import tree_leaves

            ref_leaves = tree_leaves(expected)
            exported_leaves = tree_leaves(exported.module()(*args))
            if not ref_leaves or len(ref_leaves) != len(exported_leaves):
                raise AssertionError("Exported output arity mismatch")
            for ref, out in zip(ref_leaves, exported_leaves):
                torch.testing.assert_close(out, ref, equal_nan=False)
        targets = Counter()
        schemas = {}
        for node in exported.graph.nodes:
            if node.op == "call_function" and hasattr(node.target, "_schema"):
                namespace, op, overload = str(node.target).split(".")
                key = f"{namespace}::{op}.{overload}"
                targets[key] += 1
                schemas[key] = str(node.target._schema)
        result.update(
            exported="passed",
            observed_ops=dict(sorted(targets.items())),
            observed_schemas=schemas,
            graph_count=1,
            decomposition="no user decomposition table; Buddy AOT functionalization may rewrite ops",
        )
        if name not in WORKLOADS and name not in targets:
            result.update(
                status="skipped",
                reason="Requested overload was elided or rewritten during export",
            )
            write_result(path, result)
            return result
        if mode == "trace":
            result.update(status="passed", active_stage=None)
            write_result(path, result)
            return result

        result["active_stage"] = "imported"
        write_result(path, result)
        from buddy.compiler.frontend import DynamoCompiler
        from buddy.compiler.ops import tosa

        # Cases have only explicit tensor inputs and no module parameters.
        # Compile the exact exported graph whose targets were recorded above.
        compiler = DynamoCompiler(
            primary_registry=tosa.ops_registry, enable_external_calls=False
        )
        compiler._compile_fx(exported.graph_module, list(args))
        graphs = compiler.imported_graphs
        if len(graphs) != 1:
            raise RuntimeError(f"Expected one graph, received {len(graphs)}")
        result["buddy_graph_ops"] = dict(
            Counter(type(node).__name__ for node in graphs[0].body)
        )
        result["imported"] = "passed"
        result["active_stage"] = "lowered"
        write_result(path, result)
        graphs[0].lower_to_top_level_ir()
        result["lowered"] = "passed"
        result["active_stage"] = "compiled"
        write_result(path, result)
        execute = compiler.dynamo_run()
        result["compiled"] = "passed"
        result["active_stage"] = "executed"
        write_result(path, result)
        actual = execute(*args)
        result["executed"] = "passed"
        result["active_stage"] = "correctness"
        write_result(path, result)
        compare_outputs(expected, actual, torch)
        result.update(correctness="passed", status="passed", active_stage=None)
    except Exception as exc:
        stage = result.get("active_stage")
        if stage in STAGES:
            result[stage] = "failed"
        result.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    write_result(path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root / "tests/Python/AtenOpsCoverage"))
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if request["mode"] == "environment":
        write_result(args.result, environment(request["operators"]))
    else:
        run_case(
            request["operator"],
            request["profile"],
            request["mode"],
            args.result,
        )


if __name__ == "__main__":
    main()
