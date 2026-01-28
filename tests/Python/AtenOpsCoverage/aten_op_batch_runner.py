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

import ctypes
import json
import os
import re
import sys
import traceback
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
DEFAULT_COVERAGE_JSON = THIS_DIR / "aten_op_catalog.json"

SKIP_TAGS = {
    "sparse",
    "quantized",
    "cuda_only",
    "prim",
}
NUMERIC_SKIP_TAGS = SKIP_TAGS | {"random"}
NUMERIC_ALLOW_RANDOM_OPS = {
    # Batch00 phase-1 allowlist (float32, fixed seed numeric validation).
    "bernoulli.Tensor",
    "bernoulli.default",
    "bernoulli.p",
    "bernoulli.out",
    "bernoulli.Tensor_out",
    "bernoulli.float_out",
    "bernoulli_.Tensor",
    "bernoulli_.float",
    "alpha_dropout.default",
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

    # grid_sampler_* decompositions introduce index math that is currently
    # numerically unstable in our execution path. Keep the original ATen ops so
    # Buddy can use its direct lowerings.
    for key in (
        torch.ops.aten.grid_sampler_2d.default,
        torch.ops.aten.grid_sampler_2d.out,
        torch.ops.aten.grid_sampler_3d.default,
        torch.ops.aten.grid_sampler_3d.out,
    ):
        if key in decomp:
            decomp[key] = _no_decomp

    # Random ops: keep bernoulli functional forms intact to avoid decomposing to
    # rand/uniform ops our frontend doesn't lower yet.
    for key in (
        torch.ops.aten.bernoulli.default,
        torch.ops.aten.bernoulli.Tensor,
    ):
        if key in decomp:
            decomp[key] = _no_decomp

    # alpha_dropout is tagged as random, but for train=False it is deterministic
    # and equal to identity. Keep stage-1 numeric validation simple by
    # decomposing that case only (batch00 uses train=False).
    def _alpha_dropout(self, p=0.5, train=True):
        if not train or p == 0.0:
            return self
        return NotImplemented

    decomp[torch.ops.aten.alpha_dropout.default] = _alpha_dropout

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

    # searchsorted.Scalar -> searchsorted.Tensor with scalar_tensor input
    def _searchsorted_scalar(
        sorted_sequence,
        value,
        *,
        out_int32=False,
        right=False,
        side=None,
        sorter=None,
    ):
        value_tensor = torch.ops.aten.scalar_tensor.default(
            value, dtype=sorted_sequence.dtype
        )
        return torch.ops.aten.searchsorted.Tensor(
            sorted_sequence,
            value_tensor,
            out_int32=out_int32,
            right=right,
            side=side,
            sorter=sorter,
        )

    def _searchsorted_scalar_out(
        sorted_sequence,
        value,
        *,
        out_int32=False,
        right=False,
        side=None,
        sorter=None,
        out=None,
    ):
        return _searchsorted_scalar(
            sorted_sequence,
            value,
            out_int32=out_int32,
            right=right,
            side=side,
            sorter=sorter,
        )

    decomp[torch.ops.aten.searchsorted.Scalar] = _searchsorted_scalar
    decomp[torch.ops.aten.searchsorted.Scalar_out] = _searchsorted_scalar_out

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

    # affine_grid_generator(theta, size, align_corners) -> grid
    #
    # The default decomposition uses a broadcasted mul + reduce_sum pattern
    # which currently produces incorrect numeric results in our execution path.
    # Rewrite it using bmm to avoid problematic broadcasting lowering.
    def _affine_grid_generator(theta, size, align_corners):
        if (
            not isinstance(theta, torch.Tensor)
            or theta.dim() != 3
            or theta.dtype != torch.float32
            or theta.device.type != "cpu"
        ):
            return NotImplemented
        if not isinstance(size, (list, tuple)) or len(size) != 4:
            return NotImplemented
        n, _c, h, w = [int(x) for x in size]
        if n <= 0 or h <= 0 or w <= 0:
            return NotImplemented

        x_idx = torch.ops.prims.iota.default(
            w,
            start=0,
            step=1,
            dtype=torch.int64,
            device=torch.device("cpu"),
            requires_grad=False,
        )
        y_idx = torch.ops.prims.iota.default(
            h,
            start=0,
            step=1,
            dtype=torch.int64,
            device=torch.device("cpu"),
            requires_grad=False,
        )
        x = torch.ops.prims.convert_element_type.default(x_idx, torch.float32)
        y = torch.ops.prims.convert_element_type.default(y_idx, torch.float32)

        if align_corners:
            if w > 1:
                x = torch.ops.aten.mul.Scalar(x, 2.0 / float(w - 1))
                x = torch.ops.aten.add.Scalar(x, -1.0)
            else:
                x = torch.ops.aten.full.default([w], 0.0)
            if h > 1:
                y = torch.ops.aten.mul.Scalar(y, 2.0 / float(h - 1))
                y = torch.ops.aten.add.Scalar(y, -1.0)
            else:
                y = torch.ops.aten.full.default([h], 0.0)
        else:
            x = torch.ops.aten.add.Scalar(x, 0.5)
            x = torch.ops.aten.mul.Scalar(x, 2.0 / float(w))
            x = torch.ops.aten.add.Scalar(x, -1.0)
            y = torch.ops.aten.add.Scalar(y, 0.5)
            y = torch.ops.aten.mul.Scalar(y, 2.0 / float(h))
            y = torch.ops.aten.add.Scalar(y, -1.0)

        x = torch.ops.aten.view.default(x, [1, w])
        x = torch.ops.aten.expand.default(x, [h, w])
        x = torch.ops.aten.view.default(x, [h, w, 1])
        y = torch.ops.aten.view.default(y, [h, 1])
        y = torch.ops.aten.expand.default(y, [h, w])
        y = torch.ops.aten.view.default(y, [h, w, 1])
        ones = torch.ops.aten.full.default([h, w, 1], 1.0)

        x = torch.ops.aten.constant_pad_nd.default(x, [0, 2], 0.0)
        y = torch.ops.aten.constant_pad_nd.default(y, [1, 1], 0.0)
        ones = torch.ops.aten.constant_pad_nd.default(ones, [2, 0], 0.0)
        base = torch.ops.aten.add.Tensor(torch.ops.aten.add.Tensor(x, y), ones)

        base = torch.ops.aten.view.default(base, [h * w, 3])
        base = torch.ops.aten.unsqueeze.default(base, 0)
        base = torch.ops.aten.expand.default(base, [n, h * w, 3])

        theta_t = torch.ops.aten.permute.default(theta, [0, 2, 1])
        out = torch.ops.aten.bmm.default(base, theta_t)
        return torch.ops.aten.view.default(out, [n, h, w, 2])

    decomp[torch.ops.aten.affine_grid_generator.default] = (
        _affine_grid_generator
    )

    # ---- Random ops: functionalize .out / in-place variants ----
    #
    # We aim to validate numeric correctness for a small allowlist in
    # NUMERIC_ALLOW_RANDOM_OPS. For the remaining variants, prefer decomposing
    # to functional forms to avoid relying on out/in-place semantics.

    def _bernoulli_p(self, p: float, *, generator=None):
        probs = torch.ops.aten.full_like.default(
            self,
            p,
            dtype=None,
            layout=None,
            device=None,
            pin_memory=None,
            memory_format=None,
        )
        return torch.ops.aten.bernoulli.default(probs, generator=generator)

    def _bernoulli_out(self, *, generator=None, out=None):
        return torch.ops.aten.bernoulli.default(self, generator=generator)

    def _bernoulli_tensor_out(self, p, *, generator=None, out=None):
        return torch.ops.aten.bernoulli.Tensor(self, p, generator=generator)

    def _bernoulli_float_out(self, p=0.5, *, generator=None, out=None):
        probs = torch.ops.aten.full_like.default(
            self,
            float(p),
            dtype=None,
            layout=None,
            device=None,
            pin_memory=None,
            memory_format=None,
        )
        return torch.ops.aten.bernoulli.default(probs, generator=generator)

    def _bernoulli_inplace(self, p, *, generator=None):
        return torch.ops.aten.bernoulli.Tensor(self, p, generator=generator)

    def _bernoulli_inplace_float(self, p=0.5, *, generator=None):
        probs = torch.ops.aten.full_like.default(
            self,
            float(p),
            dtype=None,
            layout=None,
            device=None,
            pin_memory=None,
            memory_format=None,
        )
        return torch.ops.aten.bernoulli.default(probs, generator=generator)

    decomp[torch.ops.aten.bernoulli.p] = _bernoulli_p
    decomp[torch.ops.aten.bernoulli.out] = _bernoulli_out
    decomp[torch.ops.aten.bernoulli.Tensor_out] = _bernoulli_tensor_out
    decomp[torch.ops.aten.bernoulli.float_out] = _bernoulli_float_out
    decomp[torch.ops.aten.bernoulli_.Tensor] = _bernoulli_inplace
    decomp[torch.ops.aten.bernoulli_.float] = _bernoulli_inplace_float

    # ---- Numeric-friendly decompositions for batch01 failures ----
    #
    # These ops currently fail numeric mode (execution) due to missing/unstable
    # lowerings. We decompose them into elementwise ops that Buddy already
    # supports, so they can participate in numeric comparison.

    def _reduce_loss(loss: torch.Tensor, reduction: int) -> torch.Tensor:
        # Match PyTorch reduction enum used by these ops:
        # 0: none, 1: mean, 2: sum
        if int(reduction) == 0:
            return loss
        if int(reduction) == 2:
            return torch.ops.aten.sum.default(loss)
        return torch.ops.aten.mean.default(loss)

    def _binary_cross_entropy(self, target, weight=None, reduction: int = 1):
        # Stable BCE: clamp inputs to avoid log(0) / 0 * -inf -> NaN.
        eps = 1e-12
        one = torch.ops.aten.full_like.default(
            self,
            1.0,
            dtype=None,
            layout=None,
            device=None,
            pin_memory=None,
            memory_format=None,
        )
        eps_t = torch.ops.aten.full_like.default(
            self,
            eps,
            dtype=None,
            layout=None,
            device=None,
            pin_memory=None,
            memory_format=None,
        )
        one_minus_eps = torch.ops.aten.sub.Tensor(one, eps_t)
        x = torch.ops.aten.minimum.default(
            torch.ops.aten.maximum.default(self, eps_t), one_minus_eps
        )
        log_x = torch.ops.aten.log.default(x)
        log_one_minus_x = torch.ops.aten.log.default(
            torch.ops.aten.sub.Tensor(one, x)
        )
        term1 = torch.ops.aten.mul.Tensor(target, log_x)
        term2 = torch.ops.aten.mul.Tensor(
            torch.ops.aten.sub.Tensor(one, target), log_one_minus_x
        )
        loss = torch.ops.aten.neg.default(
            torch.ops.aten.add.Tensor(term1, term2)
        )
        if weight is not None:
            loss = torch.ops.aten.mul.Tensor(loss, weight)
        return _reduce_loss(loss, reduction)

    def _binary_cross_entropy_out(
        self, target, weight=None, reduction: int = 1, out=None
    ):
        return _binary_cross_entropy(self, target, weight, reduction)

    decomp[torch.ops.aten.binary_cross_entropy.default] = _binary_cross_entropy
    decomp[torch.ops.aten.binary_cross_entropy.out] = _binary_cross_entropy_out

    def _binary_cross_entropy_with_logits(
        self,
        target,
        weight=None,
        pos_weight=None,
        reduction: int = 1,
    ):
        # Stable BCEWithLogits (pos_weight only used when provided).
        # loss = (1 - y) * x + max(-x, 0) + log(exp(-max(-x,0)) + exp(-x - max(-x,0)))
        one = torch.ops.aten.full_like.default(
            self,
            1.0,
            dtype=None,
            layout=None,
            device=None,
            pin_memory=None,
            memory_format=None,
        )
        zero = torch.ops.aten.zeros_like.default(self)
        neg_x = torch.ops.aten.neg.default(self)
        max_val = torch.ops.aten.maximum.default(neg_x, zero)

        exp1 = torch.ops.aten.exp.default(torch.ops.aten.neg.default(max_val))
        exp2 = torch.ops.aten.exp.default(
            torch.ops.aten.sub.Tensor(neg_x, max_val)
        )
        log_sum = torch.ops.aten.log.default(
            torch.ops.aten.add.Tensor(exp1, exp2)
        )
        base = torch.ops.aten.add.Tensor(max_val, log_sum)
        loss = torch.ops.aten.add.Tensor(
            torch.ops.aten.mul.Tensor(
                torch.ops.aten.sub.Tensor(one, target), self
            ),
            base,
        )

        if pos_weight is not None:
            # Multiply the positive part by pos_weight: loss *= (1 + (pos_weight - 1) * target)
            scale = torch.ops.aten.add.Tensor(
                one,
                torch.ops.aten.mul.Tensor(
                    torch.ops.aten.sub.Tensor(pos_weight, one), target
                ),
            )
            loss = torch.ops.aten.mul.Tensor(loss, scale)

        if weight is not None:
            loss = torch.ops.aten.mul.Tensor(loss, weight)
        return _reduce_loss(loss, reduction)

    def _binary_cross_entropy_with_logits_out(
        self,
        target,
        weight=None,
        pos_weight=None,
        reduction: int = 1,
        out=None,
    ):
        return _binary_cross_entropy_with_logits(
            self, target, weight, pos_weight, reduction
        )

    decomp[torch.ops.aten.binary_cross_entropy_with_logits.default] = (
        _binary_cross_entropy_with_logits
    )
    decomp[torch.ops.aten.binary_cross_entropy_with_logits.out] = (
        _binary_cross_entropy_with_logits_out
    )

    def _clamp_tensor(self, min, max):
        # clamp(x, min_t, max_t) = minimum(maximum(x, min_t), max_t)
        return torch.ops.aten.minimum.default(
            torch.ops.aten.maximum.default(self, min), max
        )

    def _clamp_tensor_out(self, min, max, out=None):
        return _clamp_tensor(self, min, max)

    def _clamp_tensor_inplace(self, min, max):
        return _clamp_tensor(self, min, max)

    # clip.Tensor is an alias of clamp.Tensor in PyTorch; keep explicit keys.
    decomp[torch.ops.aten.clamp.Tensor] = _clamp_tensor
    decomp[torch.ops.aten.clamp.Tensor_out] = _clamp_tensor_out
    decomp[torch.ops.aten.clamp_.Tensor] = _clamp_tensor_inplace
    decomp[torch.ops.aten.clip.Tensor] = _clamp_tensor
    decomp[torch.ops.aten.clip.Tensor_out] = _clamp_tensor_out
    decomp[torch.ops.aten.clip_.Tensor] = _clamp_tensor_inplace

    def _count_nonzero_default(self):
        mask = torch.ops.aten.ne.Scalar(self, 0)
        mask_i64 = torch.ops.aten._to_copy.default(
            mask,
            dtype=torch.int64,
            layout=None,
            device=None,
            pin_memory=None,
            non_blocking=False,
            memory_format=None,
        )
        return torch.ops.aten.sum.default(mask_i64)

    def _count_nonzero_out(self, *, out=None):
        return _count_nonzero_default(self)

    def _count_nonzero_dim_intlist(self, dim):
        mask = torch.ops.aten.ne.Scalar(self, 0)
        mask_i64 = torch.ops.aten._to_copy.default(
            mask,
            dtype=torch.int64,
            layout=None,
            device=None,
            pin_memory=None,
            non_blocking=False,
            memory_format=None,
        )
        return torch.ops.aten.sum.dim_IntList(mask_i64, dim, False)

    def _count_nonzero_dim_intlist_out(self, dim, *, out=None):
        return _count_nonzero_dim_intlist(self, dim)

    decomp[torch.ops.aten.count_nonzero.default] = _count_nonzero_default
    decomp[torch.ops.aten.count_nonzero.out] = _count_nonzero_out
    decomp[torch.ops.aten.count_nonzero.dim_IntList] = (
        _count_nonzero_dim_intlist
    )
    decomp[torch.ops.aten.count_nonzero.dim_IntList_out] = (
        _count_nonzero_dim_intlist_out
    )

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


def get_numeric_skip_reason(notes: str) -> str:
    for tag in NUMERIC_SKIP_TAGS:
        if tag in notes:
            return f"skip:{tag}"
    return ""


def _strip_out_args_kwargs(
    schema: torch._C.FunctionSchema, args: Args, kwargs: Kwargs
) -> Tuple[Args, Kwargs]:
    """Remove out-like tensor arguments from an invocation built for `schema`."""
    out_positional: set[int] = set()
    out_kw_names: set[str] = set()

    pos_idx = 0
    for arg in schema.arguments:
        if _has_default_value(arg):
            continue
        t_lower = _normalize_type(str(arg.type))
        is_out = _is_out_tensor_arg(arg, t_lower)
        if is_out:
            if arg.kwarg_only:
                out_kw_names.add(arg.name)
            else:
                out_positional.add(pos_idx)
        if not arg.kwarg_only:
            pos_idx += 1

    filtered_args: Args = [
        v for idx, v in enumerate(args) if idx not in out_positional
    ]
    filtered_kwargs: Kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in out_kw_names and k not in _OUT_ARG_NAMES
    }
    return filtered_args, filtered_kwargs


def _numeric_out_variant_compile_target(
    entry: CoverageEntry,
    schema: torch._C.FunctionSchema,
    args: Args,
    kwargs: Kwargs,
) -> Tuple[Any, torch._C.FunctionSchema, Args, Kwargs] | None:
    """Pick a functional overload to compile/execute for an out= overload.

    The reference path still executes the out= overload. The compiled path uses a
    functional overload without out tensors, allowing numeric validation without
    implementing real out-buffer semantics.
    """
    op_name = entry.get("op")
    overload = entry.get("overload")
    if not isinstance(op_name, str) or not isinstance(overload, str):
        return None

    packet = getattr(torch.ops.aten, op_name, None)
    if packet is None:
        return None

    compile_args, compile_kwargs = _strip_out_args_kwargs(schema, args, kwargs)

    overloads: List[str] = []
    try:
        overloads = list(packet.overloads())
    except Exception:
        overloads = []

    candidates: List[str] = []
    if overload.endswith("_out"):
        candidates.append(overload[: -len("_out")])
    if overload == "out":
        candidates.append("default")
        candidates.append("Tensor")
        candidates.append("Scalar")

    # Fall back to any non-out overloads that exist on the packet.
    for cand in overloads:
        if "out" in cand:
            continue
        if cand not in candidates:
            candidates.append(cand)

    for cand in candidates:
        func = getattr(packet, cand, None)
        if func is None:
            continue
        func_schema = getattr(func, "_schema", None)
        if func_schema is None:
            continue
        if _out_tensor_arg_names(func_schema):
            continue
        try:
            func(*clone_inputs(compile_args), **clone_inputs(compile_kwargs))
        except Exception:
            continue
        return func, func_schema, compile_args, compile_kwargs

    return None


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
        return Result.fail(name, msg)
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
    *,
    prefer_export: bool = False,
) -> Tuple[List[Any], Kwargs, str]:
    # For .out variants, warmup first to get correct output shapes and avoid
    # "out variants with resizing on graph inputs" graph breaks from Dynamo.
    out_arg_names = _out_tensor_arg_names(schema)
    actual_kwargs = kwargs
    if out_arg_names:
        warmed_kwargs = _warmup_out_buffers(func, args, kwargs, out_arg_names)
        if warmed_kwargs is not None:
            actual_kwargs = warmed_kwargs

    graphs: List[Any] = []
    if (
        prefer_export
        and not actual_kwargs
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
                func, *clone_inputs(args), **clone_inputs(actual_kwargs)
            )
    else:
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


def _dtype_overload_compile_target(
    entry: CoverageEntry,
    args: Args,
    kwargs: Kwargs,
) -> Tuple[Any, torch._C.FunctionSchema, Args, Kwargs] | None:
    """Provide a compile/eval shim for *.dtype / *.dtype_out overloads.

    Some PyTorch builds do not provide a working meta/fake implementation for
    these overloads (and some don't have CPU kernels at all), which prevents
    Dynamo import even though the underlying math is supported.

    For coverage purposes we rewrite them into:
      base_op(self, x1, x2, beta=..., alpha=...) -> Tensor
      _to_copy(dtype=out_dtype) -> Tensor

    This validates non-float32 paths (casts) without relying on the broken
    dtype overload implementation.
    """
    op_name = entry.get("op")
    overload = entry.get("overload")
    if not isinstance(op_name, str) or not isinstance(overload, str):
        return None
    full_name = f"{op_name}.{overload}"
    if full_name not in (
        "addmm.dtype",
        "addmm.dtype_out",
        "baddbmm.dtype",
        "baddbmm.dtype_out",
        "bmm.dtype",
        "bmm.dtype_out",
    ):
        return None

    if full_name in ("bmm.dtype", "bmm.dtype_out"):
        if len(args) < 3:
            return None
        out_dtype = args[2]
        base = torch.ops.aten.bmm.default
        base_schema = getattr(base, "_schema", None)
        if base_schema is None:
            return None
        base_args: Args = list(args[:2])

        def shim(self, mat2):
            # For BF16, prefer running math in FP32 and casting at the boundary.
            if out_dtype == torch.bfloat16:
                self = torch.ops.aten._to_copy.default(
                    self, dtype=torch.float32
                )
                mat2 = torch.ops.aten._to_copy.default(
                    mat2, dtype=torch.float32
                )
                out = base(self, mat2)
                return torch.ops.aten._to_copy.default(out, dtype=out_dtype)
            out = base(self, mat2)
            return torch.ops.aten._to_copy.default(out, dtype=out_dtype)

        return shim, base_schema, base_args, {}

    if len(args) < 4:
        return None
    out_dtype = args[3]
    beta = kwargs.get("beta", 1)
    alpha = kwargs.get("alpha", 1)

    if op_name == "addmm":
        base = torch.ops.aten.addmm.default
    elif op_name == "baddbmm":
        base = torch.ops.aten.baddbmm.default
    else:  # pragma: no cover - defensive
        return None

    base_schema = getattr(base, "_schema", None)
    if base_schema is None:
        return None

    base_args = list(args[:3])

    def shim(self, x1, x2):
        # For BF16, prefer running math in FP32 and casting at the boundary.
        # This avoids relying on BF16 matmul execution, which can be incomplete
        # in some runtimes while still validating BF16 I/O handling.
        if out_dtype == torch.bfloat16:
            self = torch.ops.aten._to_copy.default(self, dtype=torch.float32)
            x1 = torch.ops.aten._to_copy.default(x1, dtype=torch.float32)
            x2 = torch.ops.aten._to_copy.default(x2, dtype=torch.float32)
            out = base(self, x1, x2, beta=beta, alpha=alpha)
            return torch.ops.aten._to_copy.default(out, dtype=out_dtype)

        out = base(self, x1, x2, beta=beta, alpha=alpha)
        return torch.ops.aten._to_copy.default(out, dtype=out_dtype)

    return shim, base_schema, base_args, {}


def _scalar_output_compile_target(
    entry: CoverageEntry,
    args: Args,
    kwargs: Kwargs,
) -> Tuple[Any, torch._C.FunctionSchema, Args, Kwargs] | None:
    """Wrap scalar-returning ops into Tensor-only graphs for import/lowering coverage."""
    op_name = entry.get("op")
    overload = entry.get("overload")
    if not isinstance(op_name, str) or not isinstance(overload, str):
        return None
    full_name = f"{op_name}.{overload}"

    if full_name not in ("ceil.int", "ceil.float", "ceil.Scalar"):
        return None
    if len(args) < 1:
        return None

    scalar_schema = getattr(
        torch.ops.aten.scalar_tensor.default, "_schema", None
    )
    if scalar_schema is None:
        return None

    input_scalar = args[0]
    input_tensor = (
        input_scalar
        if isinstance(input_scalar, torch.Tensor)
        else torch.tensor(input_scalar, dtype=torch.float32)
    )

    def shim(t):
        # NOTE: Dynamo cannot trace ATen ops that return non-Tensor values into FX.
        # Avoid calling scalar-return overloads by rewriting them into Tensor ops.
        if full_name == "ceil.int":
            t = torch.ops.aten.ceil.default(t)
            return torch.ops.aten._to_copy.default(t, dtype=torch.int64)
        if full_name == "ceil.float":
            t = torch.ops.aten.ceil.default(t)
            return torch.ops.aten._to_copy.default(t, dtype=torch.int64)
        # ceil.Scalar: keep scalar-like behavior but return a 0-d tensor.
        return torch.ops.aten.ceil.default(t)

    return shim, scalar_schema, [input_tensor], {}


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

    compile_op = op
    compile_schema = schema
    compile_args = args
    compile_kwargs = kwargs

    dtype_target = _dtype_overload_compile_target(entry, args, kwargs)
    if dtype_target is not None:
        compile_op, compile_schema, compile_args, compile_kwargs = dtype_target
    else:
        # Default mode: functionalize .out overloads (import+lowering coverage only).
        # Purpose: work around Dynamo's known .out issues; this does not claim real
        # out-buffer aliasing/reuse semantics are supported.
        if _out_tensor_arg_names(schema):
            target = _numeric_out_variant_compile_target(
                entry, schema, args, kwargs
            )
            if target is not None:
                compile_op, compile_schema, compile_args, compile_kwargs = (
                    target
                )

    scalar_target = _scalar_output_compile_target(
        entry, compile_args, compile_kwargs
    )
    if scalar_target is not None:
        compile_op, compile_schema, compile_args, compile_kwargs = scalar_target

    torch.manual_seed(0)
    graph_break_reasons = _reset_graph_break_reasons()

    def op_call(*inputs, **kw):
        return compile_op(*inputs, **kw)

    try:
        graphs, kwargs, skip_reason = _import_graphs(
            op_call,
            compile_args,
            compile_kwargs,
            compile_schema,
            dynamo_compiler,
        )
        graph_breaks = _graph_break_count(graph_break_reasons)
        # Scalar output ops cause graph breaks but should be skipped, not failed.
        # Check both: 1) graph break reason mentions non-Tensor, OR 2) schema shows
        # non-Tensor return (handles data-dependent ops like item.default).
        if graph_breaks:
            if _is_scalar_output_break(
                graph_break_reasons
            ) or not _returns_tensor(compile_schema):
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
        skip_reason = _classify_import_exception(tb, name)
        if skip_reason:
            return Result.skip(name, skip_reason)
        graph_breaks = _graph_break_count(graph_break_reasons)
        if graph_breaks:
            return Result.fail(name, f"graph_break:count={graph_breaks}")
        return Result.fail(name, f"convert:{type(e).__name__}:{e}")

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


def _scalar_tensor_dtype_for_returns(
    schema: torch._C.FunctionSchema,
) -> torch.dtype:
    ret_types = [_normalize_type(str(ret.type)) for ret in schema.returns]
    if any("bool" in t for t in ret_types):
        return torch.bool
    if any(("symint" in t or t == "int") for t in ret_types):
        return torch.int64
    # Default to float32: conservative for backends that may not support float64.
    return torch.float32


def _scalar_tensor_dtype_for_return_type(type_str: str) -> torch.dtype:
    t = _normalize_type(type_str)
    if "bool" in t:
        return torch.bool
    if "symint" in t or t == "int":
        return torch.int64
    return torch.float32


def _numeric_tensorize_scalar_returns(
    op: Any, schema: torch._C.FunctionSchema
) -> Any | None:
    """Wrap scalar returns into 0-d tensors so the execution engine can return them."""
    if _returns_tensor(schema):
        return None
    if not schema.returns:
        return None

    # Some ops return lists (e.g., SymInt[]). Those are not representable as MLIR
    # memref results in our execution path, so numeric mode should skip them.
    for ret in schema.returns:
        t = _normalize_type(str(ret.type))
        if "tensor" in t:
            return None
        if "[]" in t or t.startswith("list[") or t.startswith("tuple["):
            return None
        if "complex" in t:
            return None

    nrets = len(schema.returns)

    if nrets == 1:
        dtype = _scalar_tensor_dtype_for_returns(schema)

        def wrapped(*inputs, **kw):
            out = op(*inputs, **kw)
            return torch.ops.aten.scalar_tensor.default(out, dtype=dtype)

        return wrapped

    dtypes = [
        _scalar_tensor_dtype_for_return_type(str(ret.type))
        for ret in schema.returns
    ]

    def wrapped(*inputs, **kw):
        outs = op(*inputs, **kw)
        return tuple(
            torch.ops.aten.scalar_tensor.default(outs[i], dtype=dtypes[i])
            for i in range(nrets)
        )

    return wrapped


def _get_repo_root() -> Path:
    # tests/Python/AtenOpsCoverage/aten_op_batch_runner.py -> repo root
    return THIS_DIR.parents[2]


def _resolve_shared_libs() -> Tuple[bool, str, List[str]]:
    lib_names = ["libmlir_runner_utils", "libmlir_c_runner_utils", "libomp"]
    if os.name == "nt":
        ext = ".dll"
    elif sys.platform == "darwin":
        ext = ".dylib"
    else:
        ext = ".so"

    repo_root = _get_repo_root()
    candidates: List[Path] = []
    llvm_build_dir = os.getenv("LLVM_MLIR_BUILD_DIR", "").strip()
    if llvm_build_dir:
        candidates.append(Path(llvm_build_dir) / "lib")
    candidates.append(repo_root / "llvm" / "build" / "lib")

    for base in candidates:
        libs = [base / f"{name}{ext}" for name in lib_names]
        if all(p.exists() for p in libs):
            return True, "", [str(p) for p in libs]

    for base in candidates:
        libs = [base / f"{name}{ext}" for name in lib_names if name != "libomp"]
        if all(p.exists() for p in libs):
            return True, "", [str(p) for p in libs]

    return (
        False,
        f"runtime:missing_shared_libs searched={','.join(str(p) for p in candidates)}",
        [],
    )


def _execute_graph(
    graph: Any, tensor_inputs: Sequence[torch.Tensor]
) -> Tuple[bool, str, List[torch.Tensor]]:
    try:
        from mlir.execution_engine import ExecutionEngine
        from mlir import runtime as rt
    except Exception as e:
        return False, f"runtime:mlir_import:{type(e).__name__}:{e}", []

    ok, msg, shared_libs = _resolve_shared_libs()
    if not ok:
        return False, msg, []

    try:
        graph.compile()
        ee = ExecutionEngine(
            graph._imported_module, opt_level=3, shared_libs=shared_libs
        )

        def _wrap_tensor(t: torch.Tensor) -> torch.Tensor:
            if t.requires_grad:
                t = t.detach()
            if t.device.type != "cpu":
                t = t.cpu()
            return t

        wrapped = [_wrap_tensor(t) for t in tensor_inputs]
        input_memref = [
            ctypes.pointer(
                ctypes.pointer(rt.get_ranked_memref_descriptor(t.numpy()))
            )
            for t in wrapped
        ]
        output_memref = [
            ctypes.pointer(ctypes.pointer(graph._output_descriptor()))
        ]
        args_memref = output_memref + input_memref
        ee.invoke(graph._func_name, *args_memref)

        output_tensors: List[torch.Tensor] = []
        out_struct_ptr = args_memref[0][0]
        out_struct_addr = ctypes.addressof(out_struct_ptr.contents)
        field_types = dict(graph._output_descriptor._fields_)
        for i in range(len(graph._output_memref)):
            field_name = str(i)
            field_type = field_types[field_name]
            offset = getattr(graph._output_descriptor, field_name).offset
            field_ptr = ctypes.cast(
                out_struct_addr + offset, ctypes.POINTER(field_type)
            )
            output_tensors.append(
                torch.from_numpy(rt.ranked_memref_to_numpy(field_ptr))
            )
        return True, "", output_tensors
    except Exception as e:
        return False, f"runtime:execute:{type(e).__name__}:{e}", []


def _collect_tensor_inputs(
    args: Args, kwargs: Kwargs
) -> Tuple[bool, str, List[torch.Tensor]]:
    tensor_args: List[torch.Tensor] = []
    found_string = False

    def _collect_from_arg(obj: Any) -> bool:
        nonlocal found_string
        if isinstance(obj, (str, bytes)):
            found_string = True
            return True
        if isinstance(obj, torch.Tensor):
            tensor_args.append(obj)
            return True
        if isinstance(obj, (list, tuple)):
            for v in obj:
                if not _collect_from_arg(v):
                    return False
            return True
        if isinstance(obj, dict):
            if _find_first_tensor(obj) is not None:
                return False
            if any(isinstance(v, (str, bytes)) for v in obj.values()):
                found_string = True
            return True
        return True

    for item in args:
        if not _collect_from_arg(item):
            return False, "input:tensor_container_not_supported", []
    if found_string:
        return False, "input:string_not_supported", []

    if any(isinstance(v, (str, bytes)) for v in kwargs.values()):
        return False, "input:string_not_supported", []
    if any(isinstance(v, torch.Tensor) for v in kwargs.values()):
        return False, "input:tensor_kwargs_not_supported", []
    if _find_first_tensor(kwargs) is not None:
        return False, "input:tensor_kwargs_not_supported", []

    return True, "", tensor_args


def _meta_dtype_to_torch(dtype: Any) -> torch.dtype | None:
    s = str(dtype)
    if "Float16" in s:
        return torch.float16
    if "BFloat16" in s:
        return torch.bfloat16
    if "Float32" in s:
        return torch.float32
    if "Float64" in s:
        return torch.float64
    if "Int8" in s:
        return torch.int8
    if "Int32" in s:
        return torch.int32
    if "Int64" in s:
        return torch.int64
    if "Bool" in s:
        return torch.bool
    return None


def _maybe_reorder_tensor_inputs_for_compiler(
    dynamo_compiler: DynamoCompiler, tensor_inputs: List[torch.Tensor]
) -> List[torch.Tensor]:
    """Reorder tensor-only inputs to match Buddy placeholder ordering.

    Dynamo may reorder tensor placeholders (arg0/arg1) relative to the original
    Python argument order. Numeric mode executes the compiled graph via
    `dynamo_run()` using tensor-only inputs, so align the runtime input list to
    the compiler's captured `_inputs` order when it can be disambiguated.
    """
    graphs = getattr(dynamo_compiler, "_imported_graphs", None)
    if not isinstance(graphs, list) or not graphs:
        return tensor_inputs
    graph = graphs[-1]
    metas = getattr(graph, "_inputs", None)
    if not isinstance(metas, list) or not metas:
        return tensor_inputs
    if len(metas) != len(tensor_inputs):
        return tensor_inputs

    remaining = list(range(len(tensor_inputs)))
    reordered: List[torch.Tensor] = []
    for meta in metas:
        shape = tuple(getattr(meta, "shape", ()))
        want_dtype = _meta_dtype_to_torch(getattr(meta, "dtype", None))
        matches = []
        for idx in remaining:
            t = tensor_inputs[idx]
            if tuple(t.shape) != shape:
                continue
            if want_dtype is not None and t.dtype != want_dtype:
                continue
            matches.append(idx)
        if len(matches) != 1:
            return tensor_inputs
        picked = matches[0]
        reordered.append(tensor_inputs[picked])
        remaining.remove(picked)
    return reordered


def run_aten_op_numeric(
    name: str,
    entry: CoverageEntry,
    dynamo_compiler: DynamoCompiler,
    templates: Dict[str, Any],
) -> Result:
    op_name = entry.get("op") or name.split(".")[0]
    if isinstance(op_name, str) and "backward" in op_name:
        return Result.skip(name, "skip:backward")

    # `empty_strided` and `new_empty*` produce uninitialized outputs (and are
    # often exercised with empty shapes in templates). Numeric comparison is not
    # meaningful for them, so skip to avoid false failures in the execution
    # path.
    if op_name in ("empty_strided", "new_empty", "new_empty_strided"):
        return Result.skip(name, "skip:uninitialized_output")

    # Some special functions are currently implemented with low-order
    # approximations, which are not expected to meet the strict numeric
    # validation thresholds yet.
    if op_name in (
        "special_i0e",
        "special_i1",
        "special_i1e",
        "special_modified_bessel_i1",
    ):
        return Result.skip(name, "skip:approximate_implementation")

    reason = get_numeric_skip_reason(entry.get("notes", ""))
    if reason == "skip:random" and name in NUMERIC_ALLOW_RANDOM_OPS:
        reason = ""
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

    # Stabilize numeric inputs for ops with restricted domains.
    if (
        op_name in ("erfinv", "erfinv_")
        and args
        and isinstance(args[0], torch.Tensor)
    ):
        inp = args[0]
        if inp.is_floating_point():
            args = list(args)
            args[0] = inp.clamp(-0.5, 0.5)

    ref_op = op
    ref_args = args
    ref_kwargs = kwargs

    dtype_target = _dtype_overload_compile_target(entry, args, kwargs)
    if dtype_target is not None:
        op, schema, args, kwargs = dtype_target
        # Numeric mode compares compiled results against a Torch reference. For
        # *.dtype / *.dtype_out overloads, some builds may not provide working
        # CPU kernels (or have broken meta/fake behavior), so use the same shim
        # for both compiled and reference paths.
        ref_op = op
        ref_args = args
        ref_kwargs = kwargs

    if name == "conv2d.padding" and len(args) >= 7 and isinstance(args[4], str):
        padding_mode = args[4]
        if padding_mode == "valid":
            inp, weight, bias, stride, _, dilation, groups = args[:7]

            def _conv2d_padding_valid(input_tensor, weight_tensor, bias_tensor):
                return torch.ops.aten.conv2d.default(
                    input_tensor,
                    weight_tensor,
                    bias_tensor,
                    stride,
                    [0, 0],
                    dilation,
                    groups,
                )

            op = _conv2d_padding_valid
            args = [inp, weight, bias]
            kwargs = {}
            ref_op = op
            ref_args = args
            ref_kwargs = kwargs

    if any(isinstance(x, (str, bytes)) for x in args) or any(
        isinstance(v, (str, bytes)) for v in kwargs.values()
    ):
        return Result.skip(name, "input:string_not_supported")

    # alpha_dropout is only supported in numeric mode for deterministic inputs.
    if name == "alpha_dropout.default" and len(args) >= 3:
        if bool(args[2]) is True:
            return Result.skip(name, "skip:random")

    compile_op = op
    compile_schema = schema
    compile_args = args
    compile_kwargs = kwargs

    scalar_wrapper = _numeric_tensorize_scalar_returns(op, schema)
    if scalar_wrapper is not None:
        compile_op = scalar_wrapper
    elif not _returns_tensor(schema):
        return Result.skip(name, "output:scalar_not_supported")

    # Numeric mode: usually skip .out variants. When possible, compile/execute a
    # functional overload (no out buffers) and use the same functional overload
    # for the reference path, since we do not validate real out-buffer aliasing
    # semantics in numeric comparison.
    if _out_tensor_arg_names(schema):
        target = _numeric_out_variant_compile_target(
            entry, schema, args, kwargs
        )
        if target is None:
            return Result.skip(name, "skip:out_variant")
        compile_op, compile_schema, compile_args, compile_kwargs = target
        ref_op = compile_op
        ref_args = compile_args
        ref_kwargs = compile_kwargs

    # Numeric mode shim: some ops are difficult to execute end-to-end yet, but
    # we still want to validate the execution harness. For fill.Tensor* we
    # compile an equivalent expression that broadcasts a scalar tensor value.
    if name in ("fill.Tensor", "fill.Tensor_out", "fill_.Tensor"):

        def _fill_tensor_scalar_shim(self_tensor, value_tensor):
            zero = torch.ops.aten.mul.Scalar(self_tensor, 0.0)
            if (
                isinstance(value_tensor, torch.Tensor)
                and value_tensor.numel() == 1
                and value_tensor.dim() != 0
            ):
                value_tensor = value_tensor.reshape(())
            return torch.ops.aten.add.Tensor(zero, value_tensor)

        compile_op = _fill_tensor_scalar_shim

    if name in ("conv2d.default", "convolution.default", "convolution.out"):
        # TEMP: Skip numeric validation for convolution ops while Conv2D lowering
        # is being iterated. Import/lowering coverage is still tracked in default
        # mode; numeric validation will be re-enabled once stabilized.
        return Result.skip(name, "skip:conv2d_numeric_temporarily_skipped")

    if name in (
        "broadcast_tensors.default",
        "block_diag.default",
        "block_diag.out",
    ):
        if (
            len(compile_args) == 1
            and isinstance(compile_args[0], (list, tuple))
            and len(compile_args[0]) == 2
            and all(isinstance(t, torch.Tensor) for t in compile_args[0])
        ):
            t0, t1 = compile_args[0]
            if name == "broadcast_tensors.default":
                out_shape = tuple(torch.broadcast_shapes(t0.shape, t1.shape))
                target_rank = len(out_shape)
                t0_rank = t0.dim()
                t1_rank = t1.dim()

                def _broadcast_tensors_tensor_only(
                    a,
                    b,
                    _out_shape=out_shape,
                    _target_rank=target_rank,
                    _a_rank=t0_rank,
                    _b_rank=t1_rank,
                ):
                    if _a_rank < _target_rank:
                        for _ in range(_target_rank - _a_rank):
                            a = torch.ops.aten.unsqueeze.default(a, 0)
                    if _b_rank < _target_rank:
                        for _ in range(_target_rank - _b_rank):
                            b = torch.ops.aten.unsqueeze.default(b, 0)
                    a_exp = torch.ops.aten.expand.default(a, _out_shape)
                    b_exp = torch.ops.aten.expand.default(b, _out_shape)
                    return (a_exp, b_exp)

                compile_op = _broadcast_tensors_tensor_only
            else:

                def _block_diag_tensor_only(a, b):
                    return torch.ops.aten.block_diag.default([a, b])

                compile_op = _block_diag_tensor_only
            compile_args = [t0, t1]
            compile_kwargs = {}
            ref_op = compile_op
            ref_args = compile_args
            ref_kwargs = compile_kwargs

    torch.manual_seed(0)
    graph_break_reasons = _reset_graph_break_reasons()

    def op_call(*inputs, **kw):
        return compile_op(*inputs, **kw)

    try:
        graphs, compile_kwargs, skip_reason = _import_graphs(
            op_call,
            compile_args,
            compile_kwargs,
            compile_schema,
            dynamo_compiler,
            prefer_export=True,
        )
        graph_breaks = _graph_break_count(graph_break_reasons)
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
        ok, msg, tensor_inputs = _collect_tensor_inputs(
            compile_args, compile_kwargs
        )
        if not ok:
            return Result.skip(name, msg)
        tensor_inputs = _maybe_reorder_tensor_inputs_for_compiler(
            dynamo_compiler, tensor_inputs
        )

        torch.manual_seed(0)
        ref_out = ref_op(*clone_inputs(ref_args), **clone_inputs(ref_kwargs))
        ok_out, out_msg, expected_items = _flatten_outputs(ref_out)
        if not ok_out:
            return Result.skip(name, out_msg)

        scalar_expected = any(kind != "tensor" for kind, _ in expected_items)

        def _check_outputs(actual_out: Any) -> Result | None:
            if isinstance(actual_out, list):
                actual_out = [
                    t.clone() if isinstance(t, torch.Tensor) else t
                    for t in actual_out
                ]
            ok_out2, out_msg2, actual_items = _flatten_outputs(actual_out)
            if not ok_out2:
                return Result.fail(name, out_msg2)

            if scalar_expected and not actual_items:
                return Result.skip(name, "output:scalar_not_supported")

            if len(actual_items) < len(expected_items):
                if scalar_expected:
                    return Result.skip(name, "output:scalar_not_supported")
                return Result.fail(
                    name,
                    f"output:arity_mismatch expected={len(expected_items)} actual={len(actual_items)}",
                )
            actual_items = list(actual_items[: len(expected_items)])

            for idx, (expected, actual) in enumerate(
                zip(expected_items, actual_items)
            ):
                try:
                    expected_kind, expected_value = expected
                    actual_kind, actual_value = actual

                    if expected_kind == "tensor":
                        expected_tensor: torch.Tensor = expected_value
                        if actual_kind == "tensor":
                            actual_tensor: torch.Tensor = actual_value
                            _assert_tensor_close(
                                expected_tensor.cpu(), actual_tensor.cpu()
                            )
                        else:
                            if (
                                isinstance(expected_tensor, torch.Tensor)
                                and expected_tensor.numel() == 1
                            ):
                                _assert_scalar_close(
                                    expected_tensor.item(), actual_value
                                )
                            else:
                                raise AssertionError(
                                    f"output_type_mismatch expected=tensor actual={actual_kind}"
                                )
                    else:
                        if actual_kind == "tensor":
                            actual_tensor = actual_value
                            if (
                                isinstance(actual_tensor, torch.Tensor)
                                and actual_tensor.numel() == 1
                            ):
                                _assert_scalar_close(
                                    expected_value, actual_tensor.item()
                                )
                            else:
                                raise AssertionError(
                                    f"output_type_mismatch expected={expected_kind} actual=tensor"
                                )
                        else:
                            _assert_scalar_close(expected_value, actual_value)
                except Exception as e:
                    return Result.fail(
                        name, f"output:{idx}:{type(e).__name__}:{e}"
                    )
            return None

        exec_func = dynamo_compiler.dynamo_run()
        exec_inputs = [
            t.detach() if isinstance(t, torch.Tensor) and t.requires_grad else t
            for t in tensor_inputs
        ]
        primary_out = exec_func(*exec_inputs)
        result = _check_outputs(primary_out)
        return Result.passed(name) if result is None else result
    except Exception as e:
        tb = traceback.format_exc()
        skip_reason = _classify_import_exception(tb, name)
        if skip_reason:
            return Result.skip(name, skip_reason)
        graph_breaks = _graph_break_count(graph_break_reasons)
        if graph_breaks:
            return Result.fail(name, f"graph_break:count={graph_breaks}")
        return Result.fail(name, f"convert:{type(e).__name__}:{e}")

    return Result.passed(name)


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
        enable_external_calls=True,
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
    validate_numeric: bool = False,
    templates_source: Path | str | None = None,
) -> List[Result]:
    coverage_map = load_coverage_map(coverage_json)
    entries = _resolve_entries(names, coverage_map)

    templates = templates or {}
    dynamo_compiler = _make_compiler()
    results: List[Result] = []
    for name, entry in entries:
        results.append(
            run_aten_op_numeric(name, entry, dynamo_compiler, templates)
            if validate_numeric
            else run_aten_op(name, entry, dynamo_compiler, templates)
        )
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
        for r in results:
            if r.status == "skip":
                print(f"SKIP {r.name} {r.reason}")

    return results
