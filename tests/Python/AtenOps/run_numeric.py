#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import runpy
import sys
import contextlib
import io
from pathlib import Path
from typing import Any, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEST_DIR = Path(__file__).resolve().parent


def _bootstrap_pythonpath() -> None:
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


@dataclasses.dataclass(frozen=True)
class TestResult:
    path: Path
    status: str  # pass | skip | fail
    reason: str = ""


def _dtype_tolerances(dtype: Any) -> Tuple[float, float]:
    import torch

    if dtype in (torch.float16, torch.bfloat16):
        return 1e-2, 1e-2
    if dtype == torch.complex64:
        return 1e-4, 1e-5
    if dtype == torch.complex128:
        return 1e-6, 1e-8
    if dtype == torch.float64:
        return 1e-6, 1e-8
    if dtype == torch.float32:
        return 1e-4, 1e-5
    return 0.0, 0.0


def _assert_tensor_close(expected: Any, actual: Any) -> None:
    import torch

    if not isinstance(expected, torch.Tensor) or not isinstance(
        actual, torch.Tensor
    ):
        raise AssertionError(
            f"output_type_mismatch expected={type(expected).__name__} actual={type(actual).__name__}"
        )

    expected = expected.detach().cpu()
    actual = actual.detach().cpu()

    if expected.shape != actual.shape:
        if expected.numel() == 1 and actual.numel() == 1:
            expected = expected.reshape(())
            actual = actual.reshape(())
        else:
            raise AssertionError(
                f"shape_mismatch expected={tuple(expected.shape)} actual={tuple(actual.shape)}"
            )

    if expected.dtype != actual.dtype:
        if expected.is_floating_point() and actual.is_floating_point():
            expected = expected.to(torch.float32)
            actual = actual.to(torch.float32)
        elif expected.is_complex() and actual.is_complex():
            target = (
                torch.complex128
                if expected.dtype == torch.complex128
                or actual.dtype == torch.complex128
                else torch.complex64
            )
            expected = expected.to(target)
            actual = actual.to(target)
        else:
            raise AssertionError(
                f"dtype_mismatch expected={expected.dtype} actual={actual.dtype}"
            )

    if expected.is_floating_point() or expected.is_complex():
        rtol, atol = _dtype_tolerances(expected.dtype)
        if not torch.allclose(
            actual, expected, rtol=rtol, atol=atol, equal_nan=True
        ):
            diff = (actual - expected).abs()
            finite = torch.isfinite(diff)
            if finite.any():
                max_abs = float(diff[finite].max().item())
            else:
                max_abs = float("nan")
            raise AssertionError(
                f"allclose_failed max_abs={max_abs} rtol={rtol} atol={atol}"
            )
    else:
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def _flatten_outputs(obj: Any) -> List[Any]:
    import torch

    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        out: List[Any] = []
        for item in obj:
            out.extend(_flatten_outputs(item))
        return out
    return []


def _collect_tensor_inputs(args: Sequence[Any]) -> List[Any]:
    import torch

    out: List[Any] = []
    for item in args:
        if isinstance(item, torch.Tensor):
            out.append(item)
    return out


def _same_tensor_metadata(expected: Any, actual: Any) -> Tuple[bool, str]:
    import torch

    if not isinstance(expected, torch.Tensor) or not isinstance(
        actual, torch.Tensor
    ):
        return False, "output_type_mismatch"
    if expected.shape != actual.shape:
        return (
            False,
            f"shape_mismatch expected={tuple(expected.shape)} actual={tuple(actual.shape)}",
        )
    if expected.dtype != actual.dtype:
        return (
            False,
            f"dtype_mismatch expected={expected.dtype} actual={actual.dtype}",
        )
    return True, ""


@dataclasses.dataclass
class _ImporterCall:
    compiler: Any
    model: Any
    args: Tuple[Any, ...]
    kwargs: dict[str, Any]


def _patch_importer(calls: List[_ImporterCall]):
    from buddy.compiler.frontend import DynamoCompiler

    original = DynamoCompiler.importer

    def wrapped(self, model, *args, **kwargs):
        calls.append(_ImporterCall(self, model, tuple(args), dict(kwargs)))
        return original(self, model, *args, **kwargs)

    return DynamoCompiler, original, wrapped


def _patch_compile_fx(compiled_inputs_by_id: dict[int, List[Any]]):
    from buddy.compiler.frontend import DynamoCompiler

    original = DynamoCompiler._compile_fx

    def wrapped(self, gm, inputs):
        compiled_inputs_by_id[id(self)] = list(inputs)
        return original(self, gm, inputs)

    return DynamoCompiler, original, wrapped


def _run_one(path: Path) -> TestResult:
    _bootstrap_pythonpath()

    import torch

    calls: List[_ImporterCall] = []
    compiled_inputs_by_id: dict[int, List[Any]] = {}
    DynamoCompiler, importer_original, importer_wrapped = _patch_importer(calls)
    _, compile_original, compile_wrapped = _patch_compile_fx(
        compiled_inputs_by_id
    )

    torch.manual_seed(0)
    import torch._dynamo

    torch._dynamo.reset()

    try:
        setattr(DynamoCompiler, "importer", importer_wrapped)
        setattr(DynamoCompiler, "_compile_fx", compile_wrapped)
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(
            buf_err
        ):
            runpy.run_path(str(path), run_name="__main__")
    except Exception as e:
        return TestResult(path=path, status="fail", reason=f"script:{e}")
    finally:
        setattr(DynamoCompiler, "importer", importer_original)
        setattr(DynamoCompiler, "_compile_fx", compile_original)

    if not calls:
        if path.name == "test_local_scalar_dense.py":
            out_s = buf_out.getvalue()
            required = ("tosa.const_shape", "tosa.reshape")
            return (
                TestResult(path=path, status="pass")
                if all(tok in out_s for tok in required)
                else TestResult(path=path, status="fail", reason="metadata")
            )
        return TestResult(path=path, status="fail", reason="no_importer_call")

    call = calls[-1]
    tensor_inputs = _collect_tensor_inputs(call.args)

    try:
        expected = call.model(*call.args, **call.kwargs)
    except Exception as e:
        return TestResult(path=path, status="fail", reason=f"eager:{e}")

    expected_items = _flatten_outputs(expected)

    def _compare(exec_out: Any) -> Tuple[bool, str]:
        actual_items = _flatten_outputs(exec_out)
        if len(actual_items) < len(expected_items):
            return False, (
                f"arity_mismatch expected={len(expected_items)} actual={len(actual_items)}"
            )
        actual_items = actual_items[: len(expected_items)]
        for idx, (exp, act) in enumerate(zip(expected_items, actual_items)):
            try:
                if path.name == "test_empty_strided.py":
                    ok, msg = _same_tensor_metadata(exp, act)
                    if not ok:
                        return False, f"output:{idx}:metadata:{msg}"
                    continue
                if (
                    path.name == "test_resize_enlarge.py"
                    and idx == 0
                    and tensor_inputs
                ):
                    n = int(tensor_inputs[0].numel())
                    n = min(n, int(exp.numel()), int(act.numel()))
                    _assert_tensor_close(
                        exp.reshape(-1)[:n], act.reshape(-1)[:n]
                    )
                else:
                    _assert_tensor_close(exp, act)
            except Exception as e:
                return False, f"output:{idx}:{type(e).__name__}:{e}"
        return True, ""

    try:
        exec_func = call.compiler.dynamo_run()
        compiled_inputs = compiled_inputs_by_id[id(call.compiler)]
        exec_inputs = [
            (x.detach().cpu() if isinstance(x, torch.Tensor) else x)
            for x in compiled_inputs
        ]
        actual = exec_func(*exec_inputs)
    except Exception as e:
        return TestResult(path=path, status="fail", reason=f"execute:{e}")

    ok, msg = _compare(actual)
    if not ok:
        return TestResult(path=path, status="fail", reason=msg)

    return TestResult(path=path, status="pass")


def _iter_tests(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.glob("test_*.py")))
        else:
            out.append(p)
    # Exclude helper scripts.
    out = [p for p in out if p.name != "run_numeric.py"]
    return out


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]] or [DEFAULT_TEST_DIR]
    tests = _iter_tests(paths)
    passed = skipped = failed = 0
    results: List[TestResult] = []
    for path in tests:
        result = _run_one(path)
        results.append(result)
        if result.status == "pass":
            passed += 1
        elif result.status == "skip":
            skipped += 1
        else:
            failed += 1

    print(
        f"SUMMARY pass={passed} fail={failed} skip={skipped} count={len(results)}"
    )
    for r in results:
        if r.status == "fail":
            print(f"FAIL {r.path} {r.reason}")
        elif r.status == "skip":
            print(f"SKIP {r.path} {r.reason}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
