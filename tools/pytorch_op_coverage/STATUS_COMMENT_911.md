# Issue #911 status comment plan (FranklinNexus)

Post on: https://github.com/buddy-compiler/buddy-mlir/issues/911  
When: after draft PR is pushed (or attach as draft PR description first).

---

## Suggested comment (English, for upstream)

> Hi @zhanghb97 — quick status on Pre-Task #911:
>
> ### Done
> - Defined coverage methodology + denominator **Buddy Target Op Set v0** (`docs/PytorchOpCoverage.md`, `tools/pytorch_op_coverage/data/target_ops_v0.json`).
> - Levels distinguished: frontend-recognized / lowered / compiled / correctness / unsupported / partial / limited.
> - MVP tool: `tools/pytorch_op_coverage/run_coverage.py` (static mode runs without a full Buddy build; `--mode live` probes DynamoCompiler import→lower when env is ready).
> - First reproducible JSON + Markdown report artifacts committed under `tools/pytorch_op_coverage/out/`.
> - Draft PR: &lt;link&gt;
>
> ### Important
> - I am **not** claiming 90% coverage yet. Static “frontend+lowering” is scaffolding evidence only; the 90% bar will use live compile+correctness on the named target set.
>
> ### Next
> 1. Wire live compile + numerical checks on Linux/WSL CI.
> 2. Expand denominator from MoE/Transformer workload traces (DeepSeek-style routing first).
> 3. Add regression microtests for highest-priority unsupported MoE ops.
>
> Happy to Request Review earlier if this MVP direction looks right.

---

## 中文备忘（自己用，不必贴到 issue）

- 先交可审查的方法论 + 工具 + 首份报告，再谈补算子。
- 不要在评论里写「已达 90%」。
- Coding phase 截止 2026-09-22；可提前 RR。
