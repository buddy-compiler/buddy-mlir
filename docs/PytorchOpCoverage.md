# PyTorch Operator Coverage Methodology (Buddy-MLIR)

This document defines how Buddy-MLIR measures PyTorch operator coverage for
[issue #911](https://github.com/buddy-compiler/buddy-mlir/issues/911).

## Pipeline under test

```text
PyTorch eager / nn.Module
        │  TorchDynamo + AOTAutograd decomp (optional inductor decomp)
        ▼
FX graph (aten / prims symbols)
        │  DynamoCompiler._ops_map   →  Buddy Graph Op classes
        ▼
Buddy Graph
        │  ops_registry (tosa / linalg / math / func / ttir)
        ▼
Top-level MLIR
        │  buddy-opt / codegen / ExecutionEngine   (live / CI)
        ▼
Compiled artifact (+ optional numerical check vs PyTorch)
```

Key sources in-tree:

| Stage | Location |
| --- | --- |
| Frontend map | `frontend/Python/frontend.py` (`DynamoCompiler._ops_map`) |
| Buddy Graph ops | `frontend/Python/graph/operation.py` |
| Lowerings | `frontend/Python/ops/{tosa,linalg,math,func,ttir,ttir_llm}.py` |
| Import examples | `examples/BuddyPython/module_gen.py`, model `import-*.py` |
| Tool | `tools/pytorch_op_coverage/` |

A frontend mapping **alone** is not full support.

## Coverage levels

| Level | Meaning | How MVP measures it |
| --- | --- | --- |
| Frontend-recognized | ATen overload key present in `_ops_map` | Static parse |
| Lowered | Buddy Graph op has an `ops_registry` entry (static), or `lower_to_top_level_ir()` succeeds (live) | Static + live |
| Compiled | Module compiles through Buddy pipeline | Live (stubbed in MVP; wire `buddy-opt` next) |
| Correctness-validated | Numerical match vs PyTorch on probe inputs | Live (stubbed in MVP) |
| Unsupported | Missing from `_ops_map` for the target key | Static |
| Partial | Mapped + lowered, but known attribute/shape/dtype limits | Annotated in target set |
| Limited shapes/dtypes | Supported only for a subset of schemas | Same as partial; expand with probe matrix later |

### Status labels emitted by the tool

- `fully_supported_static` — frontend + lowering found (evidence for scaffolding only)
- `partial` — frontend + lowering, but flagged limited
- `frontend_only` — mapped, no lowering registry entry found
- `unsupported` — not in `_ops_map`
- `live_passed` / `live_failed` / skip notes — when `--mode live`

## Denominator for the 90%+ target

**Name:** Buddy Target Op Set v0  
**File:** `tools/pytorch_op_coverage/data/target_ops_v0.json`

Rules:

1. The denominator is the set of **unique ATen overload keys** listed in v0
   (Transformer core ∪ MoE-critical ∪ ViT extras, de-duplicated).
2. Percentage claims for issue #911 must state which set version was used.
3. Until live compile+correctness is enabled in CI, reports may publish
   `fully_supported_static_pct` but **must not** claim the issue’s 90% acceptance
   criterion is met.
4. v0 is a seed. Expand by tracing representative workloads (Llama/Qwen,
   DeepSeek-MoE style routing, ViT, Whisper, embeddings) and adding newly seen
   aten keys to a versioned set (`v0.1`, `v1`, …).

### Suggested “fully supported” predicate (for the final 90% claim)

An op counts as fully supported only if:

`frontend_recognized ∧ lowered ∧ compiled ∧ correctness_validated`

Partial / limited ops count toward a separate bucket, not toward the 90%
numerator.

## MoE focus

MoE-critical families in v0 emphasize:

- routing / gating (`topk`, `softmax`, comparisons)
- expert dispatch / combine (`index*`, `gather`, `scatter*`, `masked_*`)
- dynamic tensor ops (`nonzero`, `where`, `repeat_interleave`, splits)
- expert GEMM (`mm`, `bmm`, `addmm`) and activations

Gaps in this subset are prioritized ahead of long-tail special functions.

## Reproducing the report

```bash
# Static (Windows/Linux, no Buddy build)
python tools/pytorch_op_coverage/run_coverage.py \
  --out-dir tools/pytorch_op_coverage/out

# Live (Linux/WSL recommended)
python tools/pytorch_op_coverage/run_coverage.py --mode live \
  --out-dir tools/pytorch_op_coverage/out
```

See `tools/pytorch_op_coverage/README.md` for `PYTHONPATH` setup.

## Out of scope for the first MVP PR

- Implementing every missing MoE op
- Claiming 90% coverage without live measurement
- Full model E2E as the only coverage signal (used for validation, not as the
  sole denominator)
