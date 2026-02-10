"""Common utilities for running aten coverage batches.

Features:
 - Use the coverage table (aten_coverage_catalog.json by default) to run
   DynamoCompiler import + MLIR lowering checks for a given op list.
 - Skip ops tagged in coverage notes (sparse/quantized/cuda_only/prim),
   backward ops, and ops without auto-generated inputs/graphs.
 - Mark Dynamo-uncapturable ops as skip with explicit reasons.
 - Support graph validation and numeric validation.
 - Emit SUMMARY/FAIL output for FileCheck.
"""

from __future__ import annotations

import json
import re
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

try:
    import torch._inductor.lowering  # noqa: F401
except Exception:
    # Some builds do not eagerly expose `torch._inductor.lowering` as an attribute.
    # Inductor decompositions may access it via `torch._inductor.lowering`, so we
    # import the submodule explicitly to avoid runtime AttributeError.
    pass
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_COVERAGE_JSON = THIS_DIR / "aten_coverage_catalog.json"

SKIP_TAGS = {
    "sparse",
    "quantized",
    "cuda_only",
    "prim",
}
NUMERIC_SKIP_TAGS = SKIP_TAGS
METADATA_ONLY_BASE_OPS = {
    "empty_strided",
    "empty_like",
    "new_empty",
    "new_empty_strided",
    "bernoulli",
    "bernoulli_",
    "dropout",
    "native_dropout",
    "alpha_dropout",
    "rand",
    "rand_like",
    "randn",
    "randn_like",
    "randint",
    "randint_like",
    "randperm",
    "multinomial",
    "normal",
    "normal_",
    "poisson",
    "log_normal",
    "uniform",
    "uniform_",
    "cauchy",
    "cauchy_",
    "special_airy_ai",
    "special_bessel_y0",
    "special_bessel_y1",
    "rrelu_with_noise",
    "rrelu_with_noise_",
    "rrelu_with_noise_functional",
}
INTEGER_RUNTIME_BASE_OPS = {
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "gcd",
    "lcm",
}
SHIFT_RUNTIME_BASE_OPS = {
    "bitwise_left_shift",
    "bitwise_right_shift",
}
NUMERIC_RUNTIME_UNSUPPORTED_BASE_OPS = {
    "miopen_batch_norm",
}
NUMERIC_UNCAPTURABLE_OVERLOADS = {
    "select.Dimname",
}
NUMERIC_RUNTIME_UNSUPPORTED_OVERLOADS = {
    "copysign.default",
    "div.default",
    "fmod.default",
    "frexp.default",
    "index_reduce_.default",
    "mvlgamma_.default",
    "scatter_reduce_.two",
}
NUMERIC_STRING_ARG_ALLOWED_OVERLOADS = {
    "index_reduce.default",
    "index_reduce_.default",
    "scatter_reduce.two",
    "scatter_reduce_.two",
    "segment_reduce.default",
}
NUMERIC_PRIORITY_TEMPLATE_OVERLOADS = {
    "as_strided.default",
    "as_strided_.default",
    "as_strided_copy.default",
    "empty_like.default",
    "empty_strided.default",
    "erfinv.default",
    "erfinv_.default",
    "expand.default",
    "expand_copy.default",
    "heaviside.default",
    "heaviside_.default",
    "index_put.default",
    "index_put_.default",
    "index_reduce.default",
    "index_reduce_.default",
    "logical_and.default",
    "logical_and_.default",
    "logical_or.default",
    "logical_or_.default",
    "logical_xor.default",
    "logical_xor_.default",
    "logspace.default",
    "mvlgamma.default",
    "mvlgamma_.default",
    "native_batch_norm.default",
    "new_empty_strided.default",
    "repeat.default",
    "resize.default",
    "rrelu_with_noise_functional.default",
    "scatter_reduce.two",
    "scatter_reduce_.two",
    "segment_reduce.default",
    "slice.Tensor",
    "slice_scatter.default",
    "sort.default",
    "special_airy_ai.default",
    "special_bessel_y0.default",
    "special_bessel_y1.default",
    "special_ndtri.default",
    "std.default",
    "tril_indices.default",
    "triu_indices.default",
    "var.default",
}
METADATA_SCALAR_SEMANTIC_OPS = {
    "dense_dim.default",
    "dim.default",
    "is_complex.default",
    "is_contiguous.default",
    "is_non_overlapping_and_dense.default",
    "is_pinned.default",
    "is_same_size.default",
    "numel.default",
    "size.default",
    "storage_offset.default",
    "stride.default",
    "sym_numel.default",
    "sym_size.default",
    "sym_size.int",
    "sym_storage_offset.default",
    "sym_stride.default",
    "sym_stride.int",
}
CoverageEntry = Dict[str, Any]
Args = List[Any]
Kwargs = Dict[str, Any]

_MISSING = object()
_FIXED_LIST_RE = re.compile(r"(int|symint|float|double|bool)\[(\d+)\]")
_OUT_ARG_NAMES = ("out", "values", "indices")
_ENUM_INT_DEFAULTS = {
    "dtype": torch.float32,
    "layout": torch.strided,
    "memory_format": torch.contiguous_format,
}

_DYNAMIC_OUTPUT_SHAPE_OPS = {
    "masked_select.default",
    "masked_select.out",
    "nonzero.default",
    "nonzero.out",
}


@contextmanager
def _maybe_capture_dynamic_output_shape_ops(name: str):
    """Temporarily enable Dynamo capture for dynamic-output-shape ops.

    These ops have output shapes depending on input tensor *values* (not just
    shapes). TorchDynamo treats them as dynamic-shape operators unless
    `capture_dynamic_output_shape_ops` is enabled.
    """
    cfg = getattr(torch, "_dynamo", None)
    cfg = getattr(cfg, "config", None)
    if cfg is None:
        yield
        return
    attr = "capture_dynamic_output_shape_ops"
    if not hasattr(cfg, attr):
        yield
        return

    old = getattr(cfg, attr)
    if name in _DYNAMIC_OUTPUT_SHAPE_OPS:
        setattr(cfg, attr, True)
    try:
        yield
    finally:
        setattr(cfg, attr, old)


@dataclass(frozen=True)
class Result:
    name: str
    status: str  # pass | skip | fail
    reason: str = ""

    @classmethod
    def passed(cls, name: str) -> "Result":
        return cls(name=name, status="pass")

    @classmethod
    def skip(cls, name: str, reason: str) -> "Result":
        return cls(name=name, status="skip", reason=reason)

    @classmethod
    def fail(cls, name: str, reason: str) -> "Result":
        return cls(name=name, status="fail", reason=reason)


@dataclass(frozen=True)
class BatchStats:
    passed: int
    fail: int
    skip: int

    @classmethod
    def from_results(cls, results: Iterable[Result]) -> "BatchStats":
        passed = sum(1 for r in results if r.status == "pass")
        fail = sum(1 for r in results if r.status == "fail")
        skip = sum(1 for r in results if r.status == "skip")
        return cls(passed=passed, fail=fail, skip=skip)


def _resolve_coverage_path(path: Path | str) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    repo_root = THIS_DIR.parents[2]
    for cand in (
        Path.cwd() / resolved,
        THIS_DIR / resolved,
        repo_root / resolved,
    ):
        if cand.exists():
            return cand.resolve()
    return resolved


def load_coverage_map(
    path: Path | str = DEFAULT_COVERAGE_JSON,
) -> Dict[str, CoverageEntry]:
    resolved = _resolve_coverage_path(path)
    with resolved.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    return {f"{e['op']}.{e['overload']}": e for e in entries}


def get_skip_reason(notes: str) -> str:
    for tag in SKIP_TAGS:
        if tag in notes:
            return f"skip:{tag}"
    return ""


def get_numeric_skip_reason(notes: str) -> str:
    for tag in NUMERIC_SKIP_TAGS:
        if tag in notes:
            return f"skip:{tag}"
    return ""


def _normalize_type(type_str: str) -> str:
    return type_str.replace(" ", "").lower()


def guess_value(type_str: str) -> Any:
    """Generate minimal CPU fp32 inputs from a type string."""
    t = _normalize_type(type_str)
    # Fix the RNG seed to reproduce random ops.
    torch.manual_seed(0)

    # Handle fixed-length forms like int[2] / int[3].
    m = _FIXED_LIST_RE.match(t)
    if m:
        base, num = m.group(1), int(m.group(2))
        if base in ("int", "symint"):
            return [0] * num
        if base in ("float", "double"):
            return [0.0] * num
        if base == "bool":
            return [False] * num

    # Torch schema sometimes prints List[T] (instead of T[]); normalize it here.
    if t.startswith("list[") and t.endswith("]"):
        inner = t[len("list[") : -1]
        if inner == "number":
            return [1]
        if inner in ("int", "symint"):
            return [0]
        if inner in ("float", "double"):
            return [0.0]
        if inner == "bool":
            return [False]
        if inner == "tensor" or inner.startswith("tensor"):
            return [torch.ones(1, dtype=torch.float32)]
        if inner in ("str", "string"):
            return [""]
        return None

    # For dim/dims args, return a single-dimension list or an int.
    if "int[]?" in t and "dim" in t:
        return [0]
    if "int[]?" in t and "size" in t:
        return [1]
    if "int[]" in t and "stride" in t:
        return [1]

    if "int[]" in t or "symint[]" in t:
        return [0]
    if "float[]" in t or "double[]" in t:
        return [0.0]
    if "bool[]" in t:
        return [False]
    if "scalar[]" in t:
        return [1.0]
    if "device[]" in t:
        return [torch.device("cpu")]
    if "complex" in t:
        return 0.5 + 0.1j
    if "tensor[]" in t:
        return [torch.ones(1, dtype=torch.float32)]
    if "tensor" in t:
        return torch.ones(1, dtype=torch.float32)
    if t == "number":
        return 1
    if "symint" in t or "int" in t:
        return 0
    if "float" in t or "double" in t:
        return 1.0
    if "bool" in t:
        return False
    if "scalar" in t:
        return 1.0
    if "generat" in t:
        g = torch.Generator(device="cpu")
        g.manual_seed(0)
        return g
    if "device" in t:
        return torch.device("cpu")
    if "layout" in t:
        return torch.strided
    if "memoryformat" in t:
        return torch.contiguous_format
    if "string" in t or t == "str":
        return ""
    if "dimname" in t:
        return "N"
    if "dtype" in t:
        return torch.float32
    return None


def _find_first_tensor(obj: Any) -> torch.Tensor | None:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        for x in obj:
            found = _find_first_tensor(x)
            if found is not None:
                return found
    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_first_tensor(v)
            if found is not None:
                return found
    return None


def _make_out_like(
    ref: torch.Tensor | None, dtype: torch.dtype | None = None
) -> torch.Tensor:
    # Use 0-element out buffers to avoid PyTorch deprecation warnings and
    # future errors around resizing non-empty outputs.
    if ref is not None:
        return torch.empty(0, dtype=dtype or ref.dtype, device=ref.device)
    return torch.empty(0, dtype=dtype or torch.float32)


def _has_default_value(arg: torch._C.Argument) -> bool:
    # `default_value` may be None even when a real default exists (e.g. Optional[T]=None).
    # Use has_default_value() when available to avoid treating args as required.
    if hasattr(arg, "has_default_value"):
        return arg.has_default_value()
    return arg.default_value is not None


def _is_optional_type(type_str: str, t_lower: str) -> bool:
    # PyTorch schema may encode optional as "T?" or "Optional[T]".
    return "?" in type_str or t_lower.startswith("optional[")


def _is_out_tensor_arg(arg: torch._C.Argument, t_lower: str) -> bool:
    if "tensor" not in t_lower:
        return False
    # Prefer schema aliasing info when available (covers kwarg-only out tensors
    # like `aminmax.out(min=..., max=...)`).
    if getattr(arg, "is_out", False):
        return True
    return (
        arg.name == "out"
        or arg.name.startswith("out")
        or arg.name in _OUT_ARG_NAMES
    )


def _enum_arg_default(arg: torch._C.Argument, t_lower: str) -> Any:
    # Some schemas encode enum-like args as plain ints. Avoid 0 defaults that map to
    # uint8/strided/etc and can trigger backend/fake-tensor issues.
    if t_lower in ("int", "symint") and arg.name in _ENUM_INT_DEFAULTS:
        return _ENUM_INT_DEFAULTS[arg.name]
    # Optional enum-like ints (dtype/layout/memory_format) should stay None.
    if t_lower == "optional[int]" and arg.name in _ENUM_INT_DEFAULTS:
        return None
    return _MISSING


def _infer_arg_value(
    arg: torch._C.Argument,
    type_str: str,
    t_lower: str,
    args: Args,
    kwargs: Kwargs,
) -> Any:
    if _is_out_tensor_arg(arg, t_lower):
        ref = _find_first_tensor([args, kwargs])
        target_dtype = torch.int64 if "index" in arg.name else None
        return _make_out_like(ref, dtype=target_dtype)
    enum_default = _enum_arg_default(arg, t_lower)
    if enum_default is not _MISSING:
        return enum_default

    # Prefer non-zero sizes for shape-like scalar parameters to avoid
    # generating empty tensors that can trigger backend conversion issues in
    # numeric validation (e.g., aten.eye.* with n=0).
    if t_lower in ("int", "symint") and arg.name in ("n", "m", "rows", "cols"):
        return 2

    guessed = guess_value(type_str)
    return _MISSING if guessed is None else guessed


def build_inputs(
    schema: torch._C.FunctionSchema,
) -> Tuple[bool, str, Args, Kwargs]:
    args: Args = []
    kwargs: Kwargs = {}
    for arg in schema.arguments:
        if _has_default_value(arg):
            continue
        type_str = str(arg.type)
        t_lower = _normalize_type(type_str)
        is_optional = _is_optional_type(type_str, t_lower)
        val = _infer_arg_value(arg, type_str, t_lower, args, kwargs)
        if val is _MISSING:
            if is_optional:
                val = None
            else:
                return False, f"input_gen:{arg.name}", [], {}

        if arg.kwarg_only:
            kwargs[arg.name] = val
        else:
            args.append(val)
    return True, "", args, kwargs


def _returns_tensor(schema: torch._C.FunctionSchema) -> bool:
    return any(
        "tensor" in _normalize_type(str(ret.type)) for ret in schema.returns
    )


def clone_inputs(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, (list, tuple)):
        cloned = [clone_inputs(x) for x in obj]
        return type(obj)(cloned)
    if isinstance(obj, dict):
        return {k: clone_inputs(v) for k, v in obj.items()}
    return obj


def _resolve_aten_op(entry: CoverageEntry) -> Any:
    packet = getattr(torch.ops.aten, entry["op"])
    return getattr(packet, entry["overload"])


def _canonical_base_op(op_name: str) -> str:
    return op_name[:-1] if op_name.endswith("_") else op_name


def _to_int64_runtime_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(torch.int64)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, (list, tuple)):
        converted = [_to_int64_runtime_value(v) for v in value]
        return type(value)(converted)
    if isinstance(value, dict):
        return {k: _to_int64_runtime_value(v) for k, v in value.items()}
    return value


def _normalize_shift_operand(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(torch.int64).abs()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return abs(value)
    if isinstance(value, float):
        return abs(int(value))
    if isinstance(value, (list, tuple)):
        converted = [_normalize_shift_operand(v) for v in value]
        return type(value)(converted)
    return value


def _contiguous_strides_for_size(size: List[int]) -> List[int]:
    if not size:
        return []
    strides = [1] * len(size)
    for idx in range(len(size) - 2, -1, -1):
        strides[idx] = strides[idx + 1] * max(size[idx + 1], 1)
    return strides


def _normalize_numeric_runtime_inputs(
    base_op: str,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
) -> Tuple[Args, Kwargs]:
    normalized_args: Args = list(args)
    normalized_kwargs: Kwargs = dict(kwargs)

    if base_op in INTEGER_RUNTIME_BASE_OPS:
        normalized_args = [_to_int64_runtime_value(v) for v in normalized_args]
        normalized_kwargs = {
            k: _to_int64_runtime_value(v) for k, v in normalized_kwargs.items()
        }
        if base_op in SHIFT_RUNTIME_BASE_OPS:
            if len(normalized_args) >= 2:
                normalized_args[1] = _normalize_shift_operand(
                    normalized_args[1]
                )
            if "other" in normalized_kwargs:
                normalized_kwargs["other"] = _normalize_shift_operand(
                    normalized_kwargs["other"]
                )

    if base_op == "new_empty_strided" and len(normalized_args) >= 3:
        raw_size = normalized_args[1]
        size_vals = [int(v) for v in raw_size]
        normalized_args[1] = size_vals
        normalized_args[2] = _contiguous_strides_for_size(size_vals)

    return normalized_args, normalized_kwargs


def _priority_numeric_template(
    name: str,
) -> Tuple[Args, Kwargs] | None:
    if name not in NUMERIC_PRIORITY_TEMPLATE_OVERLOADS:
        return None

    if name in ("index_reduce.default", "index_reduce_.default"):
        x = torch.zeros(3, 4, dtype=torch.float32)
        index = torch.tensor([0, 2], dtype=torch.int64)
        source = torch.ones(2, 4, dtype=torch.float32)
        return [x, 0, index, source, "mean"], {"include_self": True}

    if name in ("scatter_reduce.two", "scatter_reduce_.two"):
        x = torch.zeros(3, 4, dtype=torch.float32)
        index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.int64)
        src = torch.ones(2, 4, dtype=torch.float32)
        return [x, 0, index, src, "sum"], {"include_self": True}

    if name == "segment_reduce.default":
        data = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        lengths = torch.tensor([2, 2], dtype=torch.int64)
        return [data, "sum"], {
            "lengths": lengths,
            "axis": 0,
            "unsafe": False,
        }

    if name == "slice.Tensor":
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        return [x, 1, 0, 3, 1], {}

    if name == "slice_scatter.default":
        self_t = torch.zeros(3, 4, dtype=torch.float32)
        src = torch.ones(3, 2, dtype=torch.float32)
        return [self_t, src, 1, 1, 3, 1], {}

    if name == "sort.default":
        x = torch.randn(3, 4, dtype=torch.float32)
        return [x, 1, False], {}

    if name == "native_batch_norm.default":
        x = torch.randn(2, 3, 4, 4, dtype=torch.float32)
        weight = torch.ones(3, dtype=torch.float32)
        bias = torch.zeros(3, dtype=torch.float32)
        running_mean = torch.zeros(3, dtype=torch.float32)
        running_var = torch.ones(3, dtype=torch.float32)
        return [
            x,
            weight,
            bias,
            running_mean,
            running_var,
            True,
            0.1,
            1e-5,
        ], {}

    if name == "new_empty_strided.default":
        x = torch.zeros(1, dtype=torch.float32)
        return [x, [2, 3], [3, 1]], {}

    if name in ("index_put.default", "index_put_.default"):
        x = torch.zeros(3, 3, dtype=torch.float32)
        indices = [
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([1, 2], dtype=torch.int64),
        ]
        values = torch.tensor([1.0, 2.0], dtype=torch.float32)
        return [x, indices, values, False], {}

    if name in ("mvlgamma.default", "mvlgamma_.default"):
        x = torch.full((4,), 2.0, dtype=torch.float32)
        return [x, 2], {}

    if name in ("erfinv.default", "erfinv_.default"):
        x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32)
        return [x], {}

    if name == "special_ndtri.default":
        x = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float32)
        return [x], {}

    if name in ("std.default", "var.default"):
        x = torch.arange(8, dtype=torch.float32)
        return [x, True], {}

    if name == "special_airy_ai.default":
        x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
        return [x], {}

    if name == "special_bessel_y0.default":
        x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
        return [x], {}

    if name == "special_bessel_y1.default":
        x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
        return [x], {}

    if name == "empty_like.default":
        x = torch.randn(2, 3, dtype=torch.float32)
        return [x], {}

    if name == "rrelu_with_noise_functional.default":
        x = torch.randn(2, 3, dtype=torch.float32)
        noise = torch.zeros(2, 3, dtype=torch.float32)
        return [x, noise, 0.125, 0.3333333333333333, True, None], {}

    if name in (
        "as_strided.default",
        "as_strided_.default",
        "as_strided_copy.default",
    ):
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        return [x, [2, 2], [2, 1], 0], {}

    if name == "empty_strided.default":
        return [[2, 3], [3, 1]], {}

    if name in ("expand.default", "expand_copy.default"):
        x = torch.arange(3, dtype=torch.float32).reshape(1, 3)
        return [x, [2, 3]], {"implicit": False}

    if name == "repeat.default":
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        return [x, [2, 1]], {}

    if name == "resize.default":
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        return [x, [3, 2]], {}

    if name in ("logical_and.default", "logical_and_.default"):
        a = torch.tensor([True, False, True], dtype=torch.bool)
        b = torch.tensor([False, False, True], dtype=torch.bool)
        return [a, b], {}

    if name in ("logical_or.default", "logical_or_.default"):
        a = torch.tensor([True, False, True], dtype=torch.bool)
        b = torch.tensor([False, False, True], dtype=torch.bool)
        return [a, b], {}

    if name in ("logical_xor.default", "logical_xor_.default"):
        a = torch.tensor([True, False, True], dtype=torch.bool)
        b = torch.tensor([False, False, True], dtype=torch.bool)
        return [a, b], {}

    if name in ("heaviside.default", "heaviside_.default"):
        x = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float32)
        values = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
        return [x, values], {}

    if name == "logspace.default":
        return [0.0, 1.0, 5, 10.0], {"dtype": torch.float32}

    if name == "tril_indices.default":
        return [3, 4, 0], {"dtype": torch.int64}

    if name == "triu_indices.default":
        return [3, 4, 0], {"dtype": torch.int64}

    return None


def _get_inputs_for_op(
    name: str,
    schema: torch._C.FunctionSchema,
    templates: Dict[str, Any],
) -> Result | Tuple[Args, Kwargs]:
    if name in templates:
        try:
            args, kwargs = templates[name]()
            return args, kwargs
        except Exception as e:
            return Result.skip(name, f"template:{e}")

    priority = _priority_numeric_template(name)
    if priority is not None:
        return priority

    ok, msg, args, kwargs = build_inputs(schema)
    if not ok:
        return Result.fail(name, msg)
    return args, kwargs


def _scalar_value_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
        expected_list = [int(v) for v in expected]
        actual_list = [int(v) for v in actual]
        return expected_list == actual_list
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(expected) == bool(actual)
    return int(expected) == int(actual)


def _is_non_overlapping_and_dense_semantic(x: torch.Tensor) -> bool:
    sizes = list(x.size())
    strides = list(x.stride())
    if len(sizes) == 0:
        return True
    if any(dim == 0 for dim in sizes):
        return True

    dims = list(range(len(sizes)))
    dims.sort(key=lambda idx: strides[idx])

    expected_stride = 1
    for idx in dims:
        size = int(sizes[idx])
        stride = int(strides[idx])
        if size < 2:
            continue
        if stride != expected_stride:
            return False
        expected_stride *= size
    return True


def _metadata_scalar_expected_value(
    name: str,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
) -> Any:
    x = args[0]
    if name in ("dense_dim.default", "dim.default"):
        return int(x.dim())
    if name in ("numel.default", "sym_numel.default"):
        return int(x.numel())
    if name in ("size.default", "sym_size.default"):
        return [int(v) for v in x.size()]
    if name in ("stride.default", "sym_stride.default"):
        return [int(v) for v in x.stride()]
    if name in ("sym_size.int",):
        dim = int(kwargs.get("dim", args[1]))
        return int(x.size(dim))
    if name in ("sym_stride.int",):
        dim = int(kwargs.get("dim", args[1]))
        return int(x.stride(dim))
    if name in ("storage_offset.default", "sym_storage_offset.default"):
        return int(x.storage_offset())
    if name == "is_complex.default":
        return bool(x.is_complex())
    if name == "is_contiguous.default":
        memory_format = kwargs.get("memory_format", None)
        if memory_format is None:
            return bool(x.is_contiguous())
        return bool(x.is_contiguous(memory_format=memory_format))
    if name == "is_non_overlapping_and_dense.default":
        return _is_non_overlapping_and_dense_semantic(x)
    if name == "is_pinned.default":
        return bool(x.is_pinned())
    if name == "is_same_size.default":
        return bool(x.is_same_size(args[1]))
    raise RuntimeError(f"unsupported_metadata_scalar_semantic:{name}")


def _run_metadata_scalar_semantic_check(
    name: str,
    op: Any,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
) -> Result | None:
    if name not in METADATA_SCALAR_SEMANTIC_OPS:
        return None
    try:
        expected = _metadata_scalar_expected_value(name, args, kwargs)
        actual = op(*clone_inputs(args), **clone_inputs(kwargs))
    except Exception as e:
        return Result.fail(name, f"metadata_scalar:{type(e).__name__}:{e}")

    if _scalar_value_equal(expected, actual):
        return Result.passed(name)
    return Result.fail(
        name,
        f"metadata_scalar:mismatch expected={expected!r} actual={actual!r}",
    )


def _import_graphs(
    func: Any,
    args: Args,
    kwargs: Kwargs,
    schema: torch._C.FunctionSchema,
    compiler: DynamoCompiler,
    *,
    prefer_export: bool = False,
) -> Tuple[List[Any], Kwargs, str]:
    graphs: List[Any] = []
    if (
        prefer_export
        and not kwargs
        and all(isinstance(a, torch.Tensor) for a in args)
    ):

        class _OpModule(torch.nn.Module):
            def forward(self, *inputs):
                return func(*inputs)

        try:
            graphs = compiler.importer_by_export(
                _OpModule(), *clone_inputs(args)
            )
        except Exception:
            graphs = compiler.importer(
                func, *clone_inputs(args), **clone_inputs(kwargs)
            )
    else:
        graphs = compiler.importer(
            func, *clone_inputs(args), **clone_inputs(kwargs)
        )

    if graphs:
        return graphs, kwargs, ""
    if not _returns_tensor(schema):
        return [], kwargs, "scalar_output"
    return [], kwargs, "import_empty"


def _reset_graph_break_reasons() -> List[Any] | None:
    reasons = getattr(torch._dynamo, "graph_break_reasons", None)
    if isinstance(reasons, list):
        reasons.clear()
        return reasons
    return None


def _graph_break_count(reasons: List[Any] | None) -> int:
    if not reasons:
        return 0
    return len(reasons)


def _classify_import_exception(tb: str, op_name: str) -> str | None:
    if (
        "torch/_dynamo/variables/torch.py" in tb
        and 'assert isinstance(kwargs["out"], (TupleVariable, ListVariable))'
        in tb
    ):
        return "template:dynamo_out_overload_bug"

    if (
        (
            "torch/_functorch/_aot_autograd/functional_utils.py" in tb
            and "assert_functional_graph" in tb
        )
        or ("FunctionalizeFallbackKernel.cpp" in tb)
        or ("We only support functionalizing operators" in tb)
    ):
        return "template:functionalization_limit"

    if (
        "torch/utils/_python_dispatch.py" in tb
        and "normalize_function" in tb
        and "cannot unpack non-iterable NoneType object" in tb
    ):
        return "template:dynamo_out_overload_bug"
    return None


def run_aten_op(
    name: str,
    entry: CoverageEntry,
    dynamo_compiler: DynamoCompiler,
    templates: Dict[str, Any],
) -> Result:
    op_name = entry.get("op") or name.split(".")[0]
    if isinstance(op_name, str) and "backward" in op_name:
        return Result.skip(name, "skip:backward")

    reason = get_skip_reason(entry.get("notes", ""))
    if reason:
        return Result.skip(name, reason)

    try:
        op = _resolve_aten_op(entry)
    except Exception as e:  # pragma: no cover - defensive
        return Result.skip(name, f"lookup:{e}")

    schema = op._schema  # type: ignore[attr-defined]
    inputs = _get_inputs_for_op(name, schema, templates)
    if isinstance(inputs, Result):
        return inputs
    args, kwargs = inputs

    torch.manual_seed(0)
    graph_break_reasons = _reset_graph_break_reasons()

    def op_call(*inputs, **kw):
        return op(*inputs, **kw)

    try:
        with _maybe_capture_dynamic_output_shape_ops(name):
            graphs, _, skip_reason = _import_graphs(
                op_call,
                args,
                kwargs,
                schema,
                dynamo_compiler,
            )
    except Exception as e:
        tb = traceback.format_exc()
        classified = _classify_import_exception(tb, name)
        if classified:
            return Result.skip(name, f"dynamo_uncapturable:{classified}")
        graph_breaks = _graph_break_count(graph_break_reasons)
        if graph_breaks:
            return Result.skip(
                name,
                f"dynamo_uncapturable:graph_break:{graph_breaks}",
            )
        return Result.skip(
            name,
            f"dynamo_uncapturable:import_exception:{type(e).__name__}:{e}",
        )

    graph_breaks = _graph_break_count(graph_break_reasons)
    if graph_breaks:
        return Result.skip(
            name, f"dynamo_uncapturable:graph_break:{graph_breaks}"
        )
    if skip_reason:
        return Result.skip(name, f"dynamo_uncapturable:{skip_reason}")
    if len(graphs) != 1:
        return Result.skip(
            name, f"dynamo_uncapturable:importer_graphs={len(graphs)}"
        )

    graph = graphs[0]
    try:
        graph.lower_to_top_level_ir()
    except Exception as e:
        return Result.fail(name, f"convert:{type(e).__name__}:{e}")

    if getattr(graph, "_imported_module", None) is None:
        return Result.fail(name, "convert:empty_mlir")
    return Result.passed(name)


FlatOutputItem = Tuple[str, Any]  # ("tensor"|"float"|"int"|"bool", value)


def _flatten_outputs(obj: Any) -> Tuple[bool, str, List[FlatOutputItem]]:
    if isinstance(obj, torch.Tensor):
        return True, "", [("tensor", obj)]
    if isinstance(obj, bool):
        return True, "", [("bool", obj)]
    if isinstance(obj, int):
        return True, "", [("int", obj)]
    if isinstance(obj, float):
        return True, "", [("float", obj)]
    if isinstance(obj, complex):
        return True, "", [("complex", obj)]
    if isinstance(obj, (list, tuple)):
        out: List[FlatOutputItem] = []
        for item in obj:
            ok, msg, items = _flatten_outputs(item)
            if not ok:
                return False, msg, []
            out.extend(items)
        return True, "", out
    if obj is None:
        return False, "output:none", []
    return False, f"output:non_tensor:{type(obj).__name__}", []


def _align_tensor_outputs_by_meta(
    expected_items: List[FlatOutputItem],
    actual_items: List[FlatOutputItem],
) -> List[FlatOutputItem] | None:
    if not expected_items or not actual_items:
        return None
    if not all(
        kind == "tensor" and isinstance(value, torch.Tensor)
        for kind, value in expected_items
    ):
        return None
    if not all(
        kind == "tensor" and isinstance(value, torch.Tensor)
        for kind, value in actual_items
    ):
        return None

    candidates: List[List[int]] = []
    for _, expected_tensor in expected_items:
        expected_shape = tuple(expected_tensor.shape)
        expected_dtype = expected_tensor.dtype
        idxs: List[int] = []
        for idx, (_, actual_tensor) in enumerate(actual_items):
            if tuple(actual_tensor.shape) != expected_shape:
                continue
            if actual_tensor.dtype != expected_dtype:
                continue
            idxs.append(idx)
        if not idxs:
            return None
        candidates.append(idxs)

    used: set[int] = set()
    chosen: List[int] = []

    def _dfs(pos: int) -> bool:
        if pos == len(expected_items):
            return True
        expected_tensor = expected_items[pos][1]
        for idx in candidates[pos]:
            if idx in used:
                continue
            actual_tensor = actual_items[idx][1]
            try:
                _assert_tensor_close(expected_tensor.cpu(), actual_tensor.cpu())
            except Exception:
                continue
            used.add(idx)
            chosen.append(idx)
            if _dfs(pos + 1):
                return True
            chosen.pop()
            used.remove(idx)
        return False

    if _dfs(0):
        return [actual_items[idx] for idx in chosen]

    used.clear()
    aligned: List[FlatOutputItem] = []
    for idxs in candidates:
        picked = next((idx for idx in idxs if idx not in used), -1)
        if picked < 0:
            return None
        used.add(picked)
        aligned.append(actual_items[picked])
    return aligned


def _dtype_tolerances(dtype: torch.dtype) -> Tuple[float, float]:
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-2, 1e-2
    if dtype == torch.float64:
        return 1e-6, 1e-8
    if dtype == torch.float32:
        return 1e-4, 1e-5
    return 0.0, 0.0


def _assert_tensor_close(expected: torch.Tensor, actual: torch.Tensor) -> None:
    if expected.shape != actual.shape:
        # Treat scalar-like tensors as equivalent even if the backend returns
        # rank-1 tensors for rank-0 values.
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
            actual = actual.to(expected.dtype)
        else:
            raise AssertionError(
                f"dtype_mismatch expected={expected.dtype} actual={actual.dtype}"
            )

    if expected.is_floating_point() or expected.is_complex():
        tol_dtype = expected.dtype
        if expected.is_complex():
            tol_dtype = (
                torch.float32
                if expected.dtype == torch.complex64
                else torch.float64
            )
        rtol, atol = _dtype_tolerances(tol_dtype)
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


def _assert_tensor_metadata_equal(
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> None:
    if tuple(expected.shape) != tuple(actual.shape):
        raise AssertionError(
            f"shape_mismatch expected={tuple(expected.shape)} actual={tuple(actual.shape)}"
        )
    if expected.dtype != actual.dtype:
        raise AssertionError(
            f"dtype_mismatch expected={expected.dtype} actual={actual.dtype}"
        )


def _assert_scalar_close(expected: Any, actual: Any) -> None:
    if isinstance(expected, complex) or isinstance(actual, complex):
        if not (isinstance(expected, complex) and isinstance(actual, complex)):
            raise AssertionError(
                f"scalar_type_mismatch expected={type(expected).__name__} actual={type(actual).__name__}"
            )
        exp = complex(expected)
        act = complex(actual)
        rtol, atol = _dtype_tolerances(torch.float32)
        diff = abs(act - exp)
        if diff > (atol + rtol * abs(exp)):
            raise AssertionError(
                f"scalar_allclose_failed diff={diff} rtol={rtol} atol={atol}"
            )
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        if not (isinstance(expected, bool) and isinstance(actual, bool)):
            raise AssertionError(
                f"scalar_type_mismatch expected={type(expected).__name__} actual={type(actual).__name__}"
            )
        if expected != actual:
            raise AssertionError(
                f"scalar_mismatch expected={expected!r} actual={actual!r}"
            )
        return
    if isinstance(expected, int) and isinstance(actual, int):
        if expected != actual:
            raise AssertionError(
                f"scalar_mismatch expected={expected!r} actual={actual!r}"
            )
        return

    exp = float(expected)
    act = float(actual)
    if exp != exp or act != act:  # NaN handling without importing math.
        if exp != exp and act != act:
            return
        raise AssertionError(
            f"scalar_mismatch expected={expected!r} actual={actual!r}"
        )
    # Scalar returns in the execution path typically come back as float32 (even if
    # the Python reference is float64). Use float32 tolerances here to avoid
    # false positives.
    rtol, atol = 1e-4, 1e-5
    diff = abs(act - exp)
    if diff > (atol + rtol * abs(exp)):
        raise AssertionError(
            f"scalar_allclose_failed diff={diff} rtol={rtol} atol={atol}"
        )


def _is_metadata_only_op(name: str, entry: CoverageEntry) -> bool:
    op = str(entry.get("op") or name.split(".")[0])
    base = op[:-1] if op.endswith("_") else op
    return op in METADATA_ONLY_BASE_OPS or base in METADATA_ONLY_BASE_OPS


def _compare_output_items(
    name: str,
    expected_items: List[FlatOutputItem],
    actual_items: List[FlatOutputItem],
    *,
    metadata_only: bool,
) -> Result | None:
    if len(actual_items) < len(expected_items):
        return Result.fail(
            name,
            f"output:arity_mismatch expected={len(expected_items)} actual={len(actual_items)}",
        )
    if len(actual_items) > len(expected_items):
        aligned = _align_tensor_outputs_by_meta(
            expected_items, list(actual_items)
        )
        if aligned is not None:
            actual_items = aligned
        else:
            actual_items = list(actual_items[: len(expected_items)])
    else:
        actual_items = list(actual_items[: len(expected_items)])

    for idx, (expected, actual) in enumerate(zip(expected_items, actual_items)):
        expected_kind, expected_value = expected
        actual_kind, actual_value = actual
        try:
            if expected_kind == "tensor":
                if actual_kind != "tensor":
                    raise AssertionError(
                        f"output_type_mismatch expected=tensor actual={actual_kind}"
                    )
                expected_tensor: torch.Tensor = expected_value
                actual_tensor: torch.Tensor = actual_value
                if metadata_only:
                    _assert_tensor_metadata_equal(
                        expected_tensor, actual_tensor
                    )
                else:
                    _assert_tensor_close(
                        expected_tensor.detach().cpu(),
                        actual_tensor.detach().cpu(),
                    )
            else:
                if actual_kind != expected_kind:
                    raise AssertionError(
                        f"output_type_mismatch expected={expected_kind} actual={actual_kind}"
                    )
                if not metadata_only:
                    _assert_scalar_close(expected_value, actual_value)
        except Exception as e:
            return Result.fail(name, f"output:{idx}:{type(e).__name__}:{e}")
    return None


def _metadata_tensor_items(items: List[FlatOutputItem]) -> List[torch.Tensor]:
    return [
        value
        for kind, value in items
        if kind == "tensor" and isinstance(value, torch.Tensor)
    ]


def _assert_tensor_finite(tensor: torch.Tensor) -> None:
    if (
        tensor.is_floating_point() or tensor.is_complex()
    ) and not torch.isfinite(tensor).all():
        raise AssertionError("non_finite_values")


def _assert_tensor_integer_valued(tensor: torch.Tensor) -> None:
    if tensor.is_complex():
        raise AssertionError("complex_tensor_not_integer")
    if not tensor.is_floating_point():
        return
    if tensor.numel() == 0:
        return
    rounded = torch.round(tensor)
    if not torch.equal(tensor, rounded):
        raise AssertionError("non_integer_values")


def _assert_tensor_range(
    tensor: torch.Tensor,
    low: float,
    high: float | None,
) -> None:
    if tensor.numel() == 0:
        return
    data = tensor.detach().cpu().to(torch.float64)
    if not torch.all(data >= low):
        raise AssertionError(f"range_low_violation:{low}")
    if high is not None and not torch.all(data < high):
        raise AssertionError(f"range_high_violation:{high}")


def _randint_bounds(
    base_op: str,
    args: Sequence[Any],
) -> Tuple[float, float] | None:
    if base_op == "randint":
        if (
            len(args) >= 3
            and isinstance(args[0], int)
            and isinstance(args[1], int)
        ):
            return float(args[0]), float(args[1])
        if len(args) >= 2 and isinstance(args[0], int):
            return 0.0, float(args[0])
    if base_op == "randint_like":
        if (
            len(args) >= 3
            and isinstance(args[1], int)
            and isinstance(args[2], int)
        ):
            return float(args[1]), float(args[2])
        if len(args) >= 2 and isinstance(args[1], int):
            return 0.0, float(args[1])
    return None


def _assert_randperm_semantic(tensor: torch.Tensor, n: int) -> None:
    if tensor.dim() != 1:
        raise AssertionError("randperm_not_1d")
    if tensor.numel() != n:
        raise AssertionError(f"randperm_size_mismatch:{tensor.numel()}!={n}")
    _assert_tensor_integer_valued(tensor)
    values = tensor.to(torch.int64)
    expected = torch.arange(n, dtype=torch.int64)
    if not torch.equal(torch.sort(values).values, expected):
        raise AssertionError("randperm_not_permutation")


def _assert_dropout_mask_semantic(
    mask: torch.Tensor,
    input_tensor: torch.Tensor | None,
    *,
    train: bool,
    p: float,
) -> None:
    if mask.dtype != torch.bool:
        raise AssertionError(f"mask_dtype_mismatch:{mask.dtype}")
    if input_tensor is not None and tuple(mask.shape) != tuple(
        input_tensor.shape
    ):
        raise AssertionError(
            f"mask_shape_mismatch:{tuple(mask.shape)}!={tuple(input_tensor.shape)}"
        )

    mask_cpu = mask.detach().cpu()
    if mask_cpu.numel() == 0:
        return

    mask_as_int = mask_cpu.to(torch.int64)
    if not torch.all((mask_as_int == 0) | (mask_as_int == 1)):
        raise AssertionError("mask_non_binary")

    if not train:
        if not torch.all(mask_cpu):
            raise AssertionError("mask_not_all_true_when_eval")
        return

    if p <= 0.0:
        if not torch.all(mask_cpu):
            raise AssertionError("mask_not_all_true_when_p0")
        return

    if p >= 1.0 and torch.any(mask_cpu):
        raise AssertionError("mask_not_all_false_when_p1")


def _check_random_metadata_semantics(
    name: str,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    actual_items: List[FlatOutputItem],
) -> Result | None:
    base_op = name.split(".")[0]
    if base_op.endswith("_"):
        base_op = base_op[:-1]
    if base_op not in METADATA_ONLY_BASE_OPS:
        return None

    tensors = _metadata_tensor_items(actual_items)
    if not tensors:
        return None

    try:
        for tensor in tensors:
            _assert_tensor_finite(tensor)

        if base_op in ("rand", "rand_like"):
            for tensor in tensors:
                _assert_tensor_range(tensor, 0.0, 1.0)

        elif base_op in ("uniform",):
            low = kwargs.get("from", args[1] if len(args) > 1 else 0.0)
            high = kwargs.get("to", args[2] if len(args) > 2 else 1.0)
            low_f = float(low)
            high_f = float(high)
            for tensor in tensors:
                _assert_tensor_range(tensor, low_f, high_f)

        elif base_op in ("bernoulli",):
            for tensor in tensors:
                _assert_tensor_integer_valued(tensor)
                _assert_tensor_range(tensor, 0.0, 2.0)

        elif base_op in ("randint", "randint_like"):
            bounds = _randint_bounds(base_op, args)
            for tensor in tensors:
                _assert_tensor_integer_valued(tensor)
                if bounds is not None:
                    _assert_tensor_range(tensor, bounds[0], bounds[1])

        elif base_op in ("randperm",):
            n = int(args[0]) if args else int(tensors[0].numel())
            _assert_randperm_semantic(tensors[0], n)

        elif base_op in ("multinomial",):
            for tensor in tensors:
                _assert_tensor_integer_valued(tensor)
            if args and isinstance(args[0], torch.Tensor):
                upper = float(args[0].shape[-1])
                for tensor in tensors:
                    _assert_tensor_range(tensor, 0.0, upper)

        elif base_op in ("dropout", "native_dropout"):
            input_tensor = (
                args[0] if args and isinstance(args[0], torch.Tensor) else None
            )
            p = kwargs.get("p", args[1] if len(args) > 1 else 0.5)
            train = kwargs.get("train", args[2] if len(args) > 2 else True)
            p_f = float(p)
            train_b = bool(train)

            if base_op == "native_dropout" and len(tensors) < 2:
                raise AssertionError("native_dropout_missing_mask")

            if len(tensors) >= 2:
                _assert_dropout_mask_semantic(
                    tensors[1],
                    input_tensor,
                    train=train_b,
                    p=p_f,
                )

        elif base_op in ("poisson",):
            for tensor in tensors:
                _assert_tensor_integer_valued(tensor)
                _assert_tensor_range(tensor, 0.0, None)

    except Exception as e:
        return Result.fail(name, f"random_semantic:{type(e).__name__}:{e}")

    return None


def _run_numeric_check(
    name: str,
    compile_op: Any,
    compile_schema: torch._C.FunctionSchema,
    compile_args: Sequence[Any],
    compile_kwargs: Dict[str, Any],
    ref_op: Any,
    ref_args: Sequence[Any],
    ref_kwargs: Dict[str, Any],
    dynamo_compiler: DynamoCompiler,
    *,
    metadata_only: bool,
) -> Result:
    torch.manual_seed(0)
    graph_break_reasons = _reset_graph_break_reasons()

    def op_call(*inputs, **kw):
        return compile_op(*inputs, **kw)

    compiled_inputs_by_id: dict[int, List[Any]] = {}
    from buddy.compiler.frontend import DynamoCompiler as _DynamoCompiler

    _orig_compile_fx = _DynamoCompiler._compile_fx

    def _compile_fx_wrapped(self, gm, inputs):
        compiled_inputs_by_id[id(self)] = list(inputs)
        return _orig_compile_fx(self, gm, inputs)

    try:
        setattr(_DynamoCompiler, "_compile_fx", _compile_fx_wrapped)
        try:
            with _maybe_capture_dynamic_output_shape_ops(name):
                graphs, _, skip_reason = _import_graphs(
                    op_call,
                    compile_args,
                    compile_kwargs,
                    compile_schema,
                    dynamo_compiler,
                    prefer_export=True,
                )
        except Exception as e:
            tb = traceback.format_exc()
            classified = _classify_import_exception(tb, name)
            if classified:
                return Result.skip(name, f"dynamo_uncapturable:{classified}")
            graph_breaks = _graph_break_count(graph_break_reasons)
            if graph_breaks:
                return Result.skip(
                    name,
                    f"dynamo_uncapturable:graph_break:{graph_breaks}",
                )
            return Result.skip(
                name,
                f"dynamo_uncapturable:import_exception:{type(e).__name__}:{e}",
            )

        graph_breaks = _graph_break_count(graph_break_reasons)
        if graph_breaks:
            return Result.skip(
                name, f"dynamo_uncapturable:graph_break:{graph_breaks}"
            )
        if skip_reason:
            return Result.skip(name, f"dynamo_uncapturable:{skip_reason}")
        if len(graphs) != 1:
            return Result.skip(
                name, f"dynamo_uncapturable:importer_graphs={len(graphs)}"
            )

        compiled_inputs = compiled_inputs_by_id.get(id(dynamo_compiler), [])

        torch.manual_seed(0)
        ref_out = ref_op(*clone_inputs(ref_args), **clone_inputs(ref_kwargs))
        ok_ref, ref_msg, expected_items = _flatten_outputs(ref_out)
        if not ok_ref:
            return Result.skip(name, ref_msg)

        exec_func = dynamo_compiler.dynamo_run()
        exec_inputs = [
            t.detach() if isinstance(t, torch.Tensor) and t.requires_grad else t
            for t in compiled_inputs
        ]

        try:
            primary_out = exec_func(*exec_inputs)
        except Exception as e:
            if metadata_only:
                return Result.skip(
                    name,
                    f"metadata_only_runtime_unavailable:{type(e).__name__}:{e}",
                )
            return Result.fail(name, f"runtime:{type(e).__name__}:{e}")

        ok_out, out_msg, actual_items = _flatten_outputs(primary_out)
        if not ok_out:
            return Result.fail(name, out_msg)

        result = _compare_output_items(
            name,
            expected_items,
            actual_items,
            metadata_only=metadata_only,
        )
        if result is not None:
            return result

        if metadata_only:
            semantic_result = _check_random_metadata_semantics(
                name,
                ref_args,
                ref_kwargs,
                actual_items,
            )
            if semantic_result is not None:
                return semantic_result

        return Result.passed(name)
    except Exception as e:
        return Result.fail(name, f"numeric:{type(e).__name__}:{e}")
    finally:
        setattr(_DynamoCompiler, "_compile_fx", _orig_compile_fx)


def run_aten_coverage_numeric(
    name: str,
    entry: CoverageEntry,
    dynamo_compiler: DynamoCompiler,
    templates: Dict[str, Any],
) -> Result:
    op_name = entry.get("op") or name.split(".")[0]
    if isinstance(op_name, str) and "backward" in op_name:
        return Result.skip(name, "skip:backward")

    reason = get_numeric_skip_reason(entry.get("notes", ""))
    if reason:
        return Result.skip(name, reason)

    try:
        op = _resolve_aten_op(entry)
    except Exception as e:  # pragma: no cover - defensive
        return Result.skip(name, f"lookup:{e}")

    base_op = _canonical_base_op(str(op_name))
    if name in NUMERIC_UNCAPTURABLE_OVERLOADS:
        return Result.skip(name, "skip:dynamo_uncapturable:dimname_overload")
    if name in NUMERIC_RUNTIME_UNSUPPORTED_OVERLOADS:
        return Result.skip(name, f"skip:runtime_not_implemented:{name}")
    if base_op in NUMERIC_RUNTIME_UNSUPPORTED_BASE_OPS:
        return Result.skip(name, f"skip:runtime_not_implemented:{base_op}")

    schema = op._schema  # type: ignore[attr-defined]
    inputs = _get_inputs_for_op(name, schema, templates)
    if isinstance(inputs, Result):
        return inputs
    args, kwargs = inputs

    if any(isinstance(x, (str, bytes)) for x in args) or any(
        isinstance(v, (str, bytes)) for v in kwargs.values()
    ):
        if name not in NUMERIC_STRING_ARG_ALLOWED_OVERLOADS:
            return Result.skip(name, "input:string_not_supported")

    args, kwargs = _normalize_numeric_runtime_inputs(base_op, args, kwargs)

    scalar_semantic = _run_metadata_scalar_semantic_check(
        name,
        op,
        args,
        kwargs,
    )
    if scalar_semantic is not None:
        return scalar_semantic

    metadata_only = _is_metadata_only_op(name, entry)
    return _run_numeric_check(
        name,
        op,
        schema,
        args,
        kwargs,
        op,
        args,
        kwargs,
        dynamo_compiler,
        metadata_only=metadata_only,
    )


def _resolve_entries(
    names: Iterable[str],
    coverage_map: Dict[str, CoverageEntry],
) -> List[Tuple[str, CoverageEntry]]:
    entries: List[Tuple[str, CoverageEntry]] = []
    for name in names:
        entry = coverage_map.get(name)
        if entry is None:
            entry = {"op": name, "overload": "", "notes": "missing_in_coverage"}
        entries.append((name, entry))
    return entries


def _make_compiler() -> DynamoCompiler:
    decompositions = dict(inductor_decomp)
    return DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=decompositions,
        enable_external_calls=True,
    )


def _reset_dynamo_and_compiler() -> DynamoCompiler:
    torch._dynamo.reset()
    return _make_compiler()


def run_aten_coverage_batch(
    names: Iterable[str],
    coverage_json: Path | str = DEFAULT_COVERAGE_JSON,
    batch_label: str = "batch",
    max_fails: int = 20,
    templates: Dict[str, Any] | None = None,
    show_skips: bool = False,
    mode: str = "numeric",
) -> List[Result]:
    coverage_map = load_coverage_map(coverage_json)
    entries = _resolve_entries(names, coverage_map)

    templates = templates or {}
    results: List[Result] = []
    if mode == "numeric":
        dynamo_compiler = _make_compiler()
        for name, entry in entries:
            results.append(
                run_aten_coverage_numeric(
                    name,
                    entry,
                    dynamo_compiler,
                    templates,
                )
            )
            dynamo_compiler = _reset_dynamo_and_compiler()
    elif mode == "graph":
        dynamo_compiler = _make_compiler()
        for name, entry in entries:
            results.append(run_aten_op(name, entry, dynamo_compiler, templates))
            dynamo_compiler = _reset_dynamo_and_compiler()
    else:
        raise ValueError(f"unsupported mode: {mode}")

    stats = BatchStats.from_results(results)
    print(
        f"SUMMARY pass={stats.passed} fail={stats.fail} skip={stats.skip} "
        f"batch_label={batch_label} count={len(entries)} total={len(coverage_map)}"
    )
    print("# CHECK: SUMMARY pass=")

    remaining = max_fails
    for r in results:
        if r.status == "fail" and remaining > 0:
            print(f"FAIL {r.name} {r.reason}")
            remaining -= 1
            if remaining == 0:
                break

    if show_skips:
        for r in results:
            if r.status == "skip":
                print(f"SKIP {r.name} {r.reason}")

    return results
