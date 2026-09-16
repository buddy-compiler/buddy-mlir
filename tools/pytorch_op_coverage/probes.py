# ===- probes.py - Optional live import/lower/compile probes ---------------===
#
# Licensed under the Apache License, Version 2.0 (the "License").
# ===----------------------------------------------------------------------===
"""
Live probes for Buddy-MLIR PyTorch coverage.

Requires:
  - torch
  - buddy.compiler (BUDDY_MLIR_ENABLE_PYTHON_PACKAGES build + PYTHONPATH)
  - MLIR Python bindings on PYTHONPATH

On machines without the stack, run_coverage.py stays in static mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ProbeSpec:
    aten: str
    builder: Callable[[], tuple[Callable[..., Any], tuple[Any, ...]]]
    note: str = ""


def _try_import_stack() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return False, f"torch unavailable: {exc}"
    try:
        from buddy.compiler.frontend import DynamoCompiler  # noqa: F401
        from buddy.compiler.ops import tosa  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return False, f"buddy.compiler unavailable: {exc}"
    return True, "ok"


def seed_probes() -> dict[str, ProbeSpec]:
    """Small seed probes; expand as live env becomes available."""
    import torch

    def add():
        def fn(x, y):
            return x + y

        return fn, (torch.randn(4, 4), torch.randn(4, 4))

    def mm():
        def fn(x, y):
            return torch.mm(x, y)

        return fn, (torch.randn(4, 4), torch.randn(4, 4))

    def topk():
        def fn(x):
            return torch.topk(x, k=2)

        return fn, (torch.randn(4, 8),)

    def gather():
        def fn(x, idx):
            return torch.gather(x, 1, idx)

        return fn, (
            torch.randn(2, 8),
            torch.randint(0, 8, (2, 3)),
        )

    def scatter_add():
        def fn(x, idx, src):
            return x.scatter_add(1, idx, src)

        return fn, (
            torch.zeros(2, 8),
            torch.randint(0, 8, (2, 3)),
            torch.randn(2, 3),
        )

    def silu():
        def fn(x):
            return torch.nn.functional.silu(x)

        return fn, (torch.randn(4, 8),)

    def softmax():
        def fn(x):
            return torch.nn.functional.softmax(x, dim=-1)

        return fn, (torch.randn(4, 8),)

    return {
        "add.Tensor": ProbeSpec("add.Tensor", add),
        "mm.default": ProbeSpec("mm.default", mm),
        "topk.default": ProbeSpec("topk.default", topk, "MoE routing"),
        "gather.default": ProbeSpec("gather.default", gather, "MoE dispatch"),
        "scatter_add.default": ProbeSpec(
            "scatter_add.default", scatter_add, "MoE combine"
        ),
        "silu.default": ProbeSpec("silu.default", silu),
        "_softmax.default": ProbeSpec("_softmax.default", softmax),
    }


def run_live_probe(aten: str, probe: ProbeSpec) -> dict[str, Any]:
    """
    Import → lower_to_top_level_ir. Compilation/correctness are best-effort.

    Returns status fields consumable by classify records.
    """
    from torch._inductor.decomposition import decompositions as inductor_decomp

    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.ops import tosa

    result: dict[str, Any] = {
        "lowered": "no",
        "compiled": "not_run",
        "correctness": "not_run",
        "error": None,
        "seen_aten": [],
    }
    try:
        fn, args = probe.builder()
        compiler = DynamoCompiler(
            primary_registry=tosa.ops_registry,
            aot_autograd_decomposition=inductor_decomp,
        )
        graphs = compiler.importer(fn, *args)
        if not graphs:
            result["error"] = "importer returned no graphs"
            return result
        graph = graphs[0]
        # Collect aten-like names present after import (buddy node names differ;
        # keep FX tabular dump when verbose is needed later).
        graph.lower_to_top_level_ir()
        result["lowered"] = "yes"
        # Full buddy-opt / ExecutionEngine compile is env-specific; mark pending.
        result["compiled"] = "not_run"
        result["correctness"] = "not_run"
        result["note"] = (
            "Lowered to top-level MLIR. Compile+correctness hooks left for "
            "follow-up once buddy-opt pipeline is wired in CI."
        )
    except Exception as exc:  # pragma: no cover - depends on env
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["lowered"] = "error"
    return result
