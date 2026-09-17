# PyTorch Operator Coverage

This document describes how Buddy-MLIR measures PyTorch operator coverage
along the compilation path used by the Python frontend
([issue #911](https://github.com/buddy-compiler/buddy-mlir/issues/911)).

## Compilation path

```text
PyTorch module / function
  → TorchDynamo (+ optional AOTAutograd / Inductor decompositions)
  → FX graph (ATen / Prim symbols)
  → DynamoCompiler._ops_map  (frontend recognition)
  → Buddy Graph
  → ops_registry lowering (tosa / linalg / math / func / ttir)
  → top-level MLIR
  → compile / run  (optional correctness check vs PyTorch)
```

Relevant sources:

| Stage | Location |
| --- | --- |
| Frontend map | `frontend/Python/frontend.py` (`DynamoCompiler._ops_map`) |
| Graph ops | `frontend/Python/graph/operation.py` |
| Lowerings | `frontend/Python/ops/*.py` |
| Coverage scripts | `scripts/pytorch_op_coverage/` |

A frontend mapping alone does **not** count as full support.

## Coverage levels

| Level | Meaning |
| --- | --- |
| Frontend-recognized | ATen overload key is present in `_ops_map` |
| Lowered | Buddy Graph op has an `ops_registry` entry (static), or `lower_to_top_level_ir()` succeeds (live) |
| Compiled | Module compiles through the Buddy pipeline |
| Correctness-validated | Output matches PyTorch on probe inputs |
| Unsupported | Missing from `_ops_map` for the target key |
| Partial / limited | Mapped and lowered, but known limits on attrs, shapes, or dtypes |

For the issue **90%+** target, an operator should count as fully supported only when it is
frontend-recognized, lowered, compiled, and correctness-validated.
Partial operators are reported separately and are not counted in that numerator.

## Target operator set (denominator)

Coverage percentage is always relative to a named set:

- **Name:** Buddy Target Op Set v0
- **File:** `scripts/pytorch_op_coverage/data/target_ops_v0.json`

v0 seeds common dense-Transformer ATen symbols plus MoE-critical routing / dispatch /
index / scatter-gather ops. Expand the set from real workload traces and bump the
version (`v0.1`, `v1`, …) when the denominator changes.

Static reports may publish `fully_supported_static` (frontend map + lowering found).
That metric is **not** the same as the live 90% acceptance criterion.

## MoE focus

The MoE-critical subset prioritizes:

- routing / gating (`topk`, `softmax`, comparisons)
- expert dispatch / combine (`index*`, `gather`, `scatter*`, `masked_*`)
- dynamic tensor ops (`nonzero`, `where`, `repeat_interleave`, splits)
- expert GEMM and activations (`mm`, `bmm`, `addmm`, `silu`, `gelu`, …)

## How to reproduce

From the repository root, after the normal Buddy Python package build
(`BUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`):

```bash
# Static analysis (parses frontend sources; does not need a running Buddy build)
python scripts/pytorch_op_coverage/run_coverage.py \
  --out-dir scripts/pytorch_op_coverage/out

# Live probes (requires torch + buddy.compiler on PYTHONPATH)
export PYTHONPATH=$PWD/build/python_packages:$PWD/llvm/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH
python scripts/pytorch_op_coverage/run_coverage.py --mode live \
  --out-dir scripts/pytorch_op_coverage/out
```

Outputs:

- `pytorch_op_coverage.json` — machine-readable
- `pytorch_op_coverage.md` — human-readable summary

Live mode currently exercises a small seed probe set through import and
`lower_to_top_level_ir`. Compile and numerical checks are the next iteration.
