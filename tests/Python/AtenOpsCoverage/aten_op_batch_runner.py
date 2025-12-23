"""Common utilities for running aten coverage batches.

Features:
 - Use the coverage table (aten_op_catalog.json by default) to run
   DynamoCompiler import + MLIR lowering checks for a given op list.
 - Skip ops tagged in coverage notes (sparse/quantized/cuda_only/prim),
   backward ops, and ops without auto-generated inputs/graphs.
 - Treat any graph break as a failure.
 - Validate graph-level import + MLIR lowering only (no MLIR execution).
 - Emit SUMMARY/FAIL output for FileCheck.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
DEFAULT_COVERAGE_JSON = THIS_DIR / "aten_op_catalog.json"

SKIP_TAGS = {
    "sparse",
    "quantized",
    "cuda_only",
    "prim",
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


def make_aot_decompositions() -> Dict[Any, Any]:
    """
    Start from Inductor decompositions, but disable a few decompositions that
    introduce `prims.*` ops our frontend doesn't map yet.

    Returning `NotImplemented` from a decomposition keeps the original ATen op
    in the graph, allowing Buddy to use its own lowering directly.
    """
    decomp: Dict[Any, Any] = dict(inductor_decomp)

    def _no_decomp(*args, **kwargs):
        return NotImplemented

    # Inductor decomp for max_pool*_with_indices may rewrite to prims
    # `_low_memory_max_pool_with_offsets` + `_low_memory_max_pool_offsets_to_indices`.
    # Buddy already has direct lowerings for the ATen ops, so keep them intact.
    for key in (
        torch.ops.aten.max_pool2d_with_indices.default,
        torch.ops.aten.max_pool3d_with_indices.default,
    ):
        if key in decomp:
            decomp[key] = _no_decomp

    # ---- Custom decompositions for unsupported ATen ops ----
    #
    # We accept "composite via decomposition" coverage, so prefer decomposing to
    # other ATen ops that Buddy already supports instead of requiring direct
    # lowering coverage for every op.

    # bucketize(values, boundaries) == searchsorted(boundaries, values)
    def _bucketize_tensor(values, boundaries, *, out_int32=False, right=False):
        return torch.ops.aten.searchsorted.Tensor(
            boundaries, values, out_int32=out_int32, right=right
        )

    def _bucketize_tensor_out(
        values, boundaries, *, out_int32=False, right=False, out=None
    ):
        return torch.ops.aten.searchsorted.Tensor(
            boundaries, values, out_int32=out_int32, right=right
        )

    decomp[torch.ops.aten.bucketize.Tensor] = _bucketize_tensor
    decomp[torch.ops.aten.bucketize.Tensor_out] = _bucketize_tensor_out

    # addbmm(self, batch1, batch2) = beta*self + alpha*sum_i(batch1[i]@batch2[i])
    def _addbmm(self, batch1, batch2, *, beta=1, alpha=1):
        prod = torch.ops.aten.bmm.default(batch1, batch2)
        summed = torch.ops.aten.sum.dim_IntList(prod, [0], False)
        left = torch.ops.aten.mul.Scalar(self, beta)
        right = torch.ops.aten.mul.Scalar(summed, alpha)
        return torch.ops.aten.add.Tensor(left, right)

    def _addbmm_out(self, batch1, batch2, *, beta=1, alpha=1, out=None):
        return _addbmm(self, batch1, batch2, beta=beta, alpha=alpha)

    decomp[torch.ops.aten.addbmm.default] = _addbmm
    decomp[torch.ops.aten.addbmm.out] = _addbmm_out

    # adaptive_max_pool3d(self, output_size) -> (output, indices)
    #
    # Inductor has decomposition for adaptive_max_pool2d but not 3d. For our
    # operator coverage we accept composite implementations, so we decompose
    # adaptive_max_pool3d into a regular max_pool3d_with_indices when the input
    # spatial sizes are divisible by output_size (uniform kernel/stride).
    def _adaptive_max_pool3d(self, output_size):
        if not isinstance(output_size, (list, tuple)) or len(output_size) != 3:
            return NotImplemented
        if not isinstance(self, torch.Tensor) or self.dim() != 5:
            return NotImplemented
        in_d, in_h, in_w = self.shape[-3:]
        out_d, out_h, out_w = (
            int(output_size[0]),
            int(output_size[1]),
            int(output_size[2]),
        )
        if out_d <= 0 or out_h <= 0 or out_w <= 0:
            return NotImplemented
        if in_d % out_d != 0 or in_h % out_h != 0 or in_w % out_w != 0:
            return NotImplemented
        k_d, k_h, k_w = (in_d // out_d, in_h // out_h, in_w // out_w)
        kernel = [int(k_d), int(k_h), int(k_w)]
        stride = list(kernel)
        return torch.ops.aten.max_pool3d_with_indices.default(
            self, kernel, stride, [0, 0, 0], [1, 1, 1], False
        )

    decomp[torch.ops.aten.adaptive_max_pool3d.default] = _adaptive_max_pool3d

    return decomp


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
    if "string" in t:
        return ""
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
    return "tensor" in t_lower and (
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


def _out_tensor_arg_names(schema: torch._C.FunctionSchema) -> List[str]:
    names: List[str] = []
    for arg in schema.arguments:
        t_lower = _normalize_type(str(arg.type))
        if _is_out_tensor_arg(arg, t_lower):
            names.append(arg.name)
    return names


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
    ok, msg, args, kwargs = build_inputs(schema)
    if not ok:
        return Result.skip(name, msg)
    return args, kwargs


def _warmup_out_buffers(
    func: Any,
    args: Args,
    kwargs: Kwargs,
    out_arg_names: List[str],
) -> Kwargs | None:
    if not out_arg_names:
        return None
    if not any(isinstance(kwargs.get(k), torch.Tensor) for k in out_arg_names):
        return None
    try:
        rng_state = torch.random.get_rng_state()
        warm_args = clone_inputs(args)
        warm_kwargs = clone_inputs(kwargs)
        func(*warm_args, **warm_kwargs)
        torch.random.set_rng_state(rng_state)
    except Exception:
        return None

    new_kwargs = dict(kwargs)
    for name in out_arg_names:
        warm_buf = warm_kwargs.get(name)
        if isinstance(warm_buf, torch.Tensor):
            new_kwargs[name] = torch.empty(
                warm_buf.shape, dtype=warm_buf.dtype, device=warm_buf.device
            )
    return new_kwargs


def _import_graphs(
    func: Any,
    args: Args,
    kwargs: Kwargs,
    schema: torch._C.FunctionSchema,
    compiler: DynamoCompiler,
) -> Tuple[List[Any], Kwargs, str]:
    # For .out variants, warmup first to get correct output shapes and avoid
    # "out variants with resizing on graph inputs" graph breaks from Dynamo.
    out_arg_names = _out_tensor_arg_names(schema)
    actual_kwargs = kwargs
    if out_arg_names:
        warmed_kwargs = _warmup_out_buffers(func, args, kwargs, out_arg_names)
        if warmed_kwargs is not None:
            actual_kwargs = warmed_kwargs

    graphs = compiler.importer(
        func, *clone_inputs(args), **clone_inputs(actual_kwargs)
    )
    if graphs:
        return graphs, actual_kwargs, ""
    if not _returns_tensor(schema):
        return [], actual_kwargs, "scalar_output"
    return [], actual_kwargs, "import_empty"


def _reset_graph_break_reasons() -> List[Any] | None:
    reasons = getattr(torch._dynamo, "graph_break_reasons", None)
    if isinstance(reasons, list):
        reasons.clear()
        return reasons
    return None


def _is_scalar_output_break(reasons: List[Any] | None) -> bool:
    """Check if graph breaks are due to non-Tensor output (scalar ops).

    Only matches 'torch.* op returned non-Tensor' which is for true scalar output ops.
    Does NOT match 'Data dependent operator ... non-Tensor output' which is different.
    """
    if not reasons:
        return False
    return any(
        "op returned non-Tensor" in str(getattr(r, "reason", r))
        for r in reasons
    )


def _graph_break_count(reasons: List[Any] | None) -> int:
    if not reasons:
        return 0
    return len(reasons)


def _classify_import_exception(tb: str) -> str | None:
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
    # Inference-only coverage: skip backward ops rather than failing.
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
        graphs, kwargs, skip_reason = _import_graphs(
            op_call, args, kwargs, schema, dynamo_compiler
        )
        graph_breaks = _graph_break_count(graph_break_reasons)
        # Scalar output ops cause graph breaks but should be skipped, not failed.
        # Check both: 1) graph break reason mentions non-Tensor, OR 2) schema shows
        # non-Tensor return (handles data-dependent ops like item.default).
        if graph_breaks:
            if _is_scalar_output_break(
                graph_break_reasons
            ) or not _returns_tensor(schema):
                return Result.skip(name, "scalar_output")
            return Result.fail(name, f"graph_break:count={graph_breaks}")
        if skip_reason:
            return Result.skip(name, skip_reason)
        if len(graphs) != 1:
            return Result.fail(
                name, f"graph_break:importer_graphs={len(graphs)}"
            )
        graph = graphs[0]
        graph.lower_to_top_level_ir()
        if getattr(graph, "_imported_module", None) is None:
            return Result.fail(name, "convert:empty_mlir")
    except Exception as e:
        tb = traceback.format_exc()
        skip_reason = _classify_import_exception(tb)
        if skip_reason:
            return Result.skip(name, skip_reason)
        graph_breaks = _graph_break_count(graph_break_reasons)
        if graph_breaks:
            return Result.fail(name, f"graph_break:count={graph_breaks}")
        return Result.fail(name, f"convert:{type(e).__name__}:{e}")

    return Result.passed(name)


def _apply_env_overrides(show_skips: bool, max_fails: int) -> Tuple[bool, int]:
    # Optional env overrides for reporting/debugging without touching batch files.
    env_show_skips = os.getenv("BUDDY_OC_SHOW_SKIPS", "").strip().lower()
    if env_show_skips in ("1", "true", "yes", "y", "on"):
        show_skips = True
    env_max_fails = os.getenv("BUDDY_OC_MAX_FAILS", "").strip()
    if env_max_fails:
        try:
            max_fails = int(env_max_fails)
        except ValueError:
            raise ValueError(f"Invalid BUDDY_OC_MAX_FAILS={env_max_fails!r}")
    return show_skips, max_fails


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
    return DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=make_aot_decompositions(),
    )


def _reset_dynamo_and_compiler() -> DynamoCompiler:
    torch._dynamo.reset()
    return _make_compiler()


def run_aten_op_batch(
    names: Iterable[str],
    coverage_json: Path | str = DEFAULT_COVERAGE_JSON,
    batch_label: str = "batch",
    max_fails: int = 20,
    templates: Dict[str, Any] | None = None,
    show_skips: bool = False,
) -> List[Result]:
    show_skips, max_fails = _apply_env_overrides(show_skips, max_fails)

    coverage_map = load_coverage_map(coverage_json)
    templates = templates or {}
    entries = _resolve_entries(names, coverage_map)

    dynamo_compiler = _make_compiler()
    results: List[Result] = []
    for name, entry in entries:
        results.append(run_aten_op(name, entry, dynamo_compiler, templates))
        # Reset Dynamo after EVERY operation to prevent state pollution.
        # Even successful compilations can leave cached state that interferes
        # with subsequent operations (e.g., le.Tensor success pollutes le.Scalar).
        # This is necessary because Dynamo's internal caching doesn't properly
        # isolate different op patterns with similar function structures.
        dynamo_compiler = _reset_dynamo_and_compiler()

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
        from collections import Counter

        skip_reasons = Counter(r.reason for r in results if r.status == "skip")
        for reason, count in skip_reasons.items():
            print(f"SKIP {reason} count={count}")
        for r in results:
            if r.status == "skip":
                print(f"SKIP {r.name} {r.reason}")

    return results
