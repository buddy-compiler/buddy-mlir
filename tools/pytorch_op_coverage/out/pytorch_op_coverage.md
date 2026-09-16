# Buddy-MLIR PyTorch Operator Coverage Report

- Generated (UTC): `2026-09-16T16:10:43.295581+00:00`
- Issue: https://github.com/buddy-compiler/buddy-mlir/issues/911
- Mode: `static`
- Repo rev: `6699773`
- Target set: `C:\Temp\buddy-mlir-911\buddy-mlir\tools\pytorch_op_coverage\data\target_ops_v0.json`

## Summary

- Denominator: **Buddy Target Op Set v0 (unique aten keys)**
- Total ops: **108**
- Frontend recognized: **101** (93.52%)
- Has Buddy lowering: **101** (93.52%)
- Fully supported (static): **89** (82.41%)
- Partial / limited: **12** (11.11%)
- Frontend only (no lowering): **0**
- Unsupported: **7** (6.48%)

> fully_supported_static is NOT live compile/correctness coverage. Do not claim issue #911 90% until live mode measures compiled+correct.

## MoE-critical subset

- Total: **47**
- Fully supported (static): **35** (74.47%)
- Partial: **7**
- Unsupported: **5**

## Unsupported operators

| ATen | Family | Notes |
| --- | --- | --- |
| `argsort.default` | moe_critical | No DynamoCompiler._ops_map entry. |
| `index_add.default` | moe_critical | No DynamoCompiler._ops_map entry. |
| `index_copy.default` | moe_critical | No DynamoCompiler._ops_map entry. |
| `one_hot.default` | moe_critical | No DynamoCompiler._ops_map entry. |
| `bincount.default` | moe_critical | No DynamoCompiler._ops_map entry. |
| `pixel_shuffle.default` | vision_transformer_extra | No DynamoCompiler._ops_map entry. |
| `pixel_unshuffle.default` | vision_transformer_extra | No DynamoCompiler._ops_map entry. |

## Partial / limited operators

| ATen | Buddy op | Dialects | Limitation |
| --- | --- | --- | --- |
| `reshape.default` | `ViewOp` | tosa, linalg, ttir | FX often emits view.default; tracked via decomp_aliases. |
| `_scaled_dot_product_flash_attention_for_cpu.default` | `ScaledDotProductFlashAttentionForCpuOp` | tosa | CPU flash path; GPU/other backends may differ. |
| `index.Tensor` | `IndexOp` | linalg, ttir_llm | Advanced indexing / None / boolean masks may be partial. |
| `fill_cache.default` | `FillCacheOp` | ttir_llm | Buddy KV-cache helper; not a generic ATen public op. |
| `update_cache.default` | `UpdateCacheOp` | ttir_llm | Buddy KV-cache helper; not a generic ATen public op. |
| `contiguous.default` | `CloneOp` | tosa, linalg, ttir_llm | Often elided; treated as alias of clone.default when present. |
| `topk.default` | `TopkOp` | linalg | Often shape/k-attr limited; validate expert routing sizes. |
| `softmax.default` | `SoftmaxOp` | linalg, ttir_llm | FX often emits _softmax.default; tracked via decomp_aliases. |
| `index_put.default` | `IndexPutOp` | linalg, ttir_llm | Accumulate / advanced indexing may be partial. |
| `scatter.reduce` | `ScatterReduceOp` | linalg | Reduce mode / dtype coverage may be partial. |
| `scatter_reduce.two` | `ScatterReduceOp` | linalg | Reduce mode / dtype coverage may be partial. |
| `pad.default` | `ConstantPadNdOp` | tosa | Often lowers via constant_pad_nd; tracked via decomp_aliases. |

## Frontend-only (mapped, no lowering found)

_None._

## High-priority follow-ups

1. Enable live mode (import→lower→compile→correctness) on Linux/WSL CI.
2. Trace real MoE workloads to expand the denominator beyond v0 seed.
3. Prioritize unsupported MoE ops that block expert dispatch/combine.
4. Add regression microtests for each newly supported MoE op.

