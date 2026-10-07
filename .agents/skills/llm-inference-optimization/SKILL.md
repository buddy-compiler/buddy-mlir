---
name: llm-inference-optimization
description: Optimize LLM inference built by Buddy-MLIR (buddy-codegen models, .rax, buddy-cli) - prefill, decode, attention, KV cache, LM head, sampling, speculative decoding - and validate it with perplexity and fair llama.cpp comparisons. Use for any change to model kernels (e.g. graph/transform/k3_w4.py), model specs, the model session or runtime, or when asked how fast a model runs.
---

# LLM inference optimization

Follow the `performance-optimization` workflow; this skill adds what is
specific to LLMs.

## Know which phase you are optimizing

- **Decode** (one token per step) reads every weight once per token: it is
  DRAM-bandwidth-bound once the matmul kernels are efficient. Floor =
  weight bytes / read bandwidth. Above the floor: attention (grows with the
  context), the LM head (a large share for small models), synchronization,
  per-token runtime overhead. `references/decode.md`.
- **Prefill** (many tokens per step) is compute-bound: matrix units, tiling,
  activation reuse, data movement around the compute. `references/prefill.md`.
- Measure each at several context lengths (e.g. 64, 458, 900 tokens):
  attention costs change with the position, matmul costs do not.

## Validate

- Same text as the baseline for bit-identical changes; perplexity for
  numerics changes: `buddy-cli --model m.rax --perplexity wiki.test.ids
  --ppl-context 512 --ppl-chunks 40` (scored like llama-perplexity; ids from
  `llama-tokenize --ids` or one id per line). See `docs/Runtime.md`.
- `references/baselines.md`: comparing against llama.cpp fairly (formats,
  LM head, thread placement, perplexity).

## Platform history

Read the platform's files before proposing an optimization: they list what
was done, the measured results, and the ideas measured and rejected (with
why), so that they are not proposed again without new evidence.

| Platform | Files |
| --- | --- |
| SpacemiT K3 (`w4g32`, `docs/K3DeepSeekR1.md`) | `references/spacemit-k3-decode.md`, `references/spacemit-k3-prefill.md`, `references/spacemit-k3-rejected-ideas.md` |
| Tenstorrent | none yet |

Add a file per platform and topic (`<vendor>-<chip>-<topic>.md`) when a new
platform gets optimization work.

## Scripts

- `scripts/spacemit-k3-w4-kernel-bench.py`: builds the `w4g32` matmul
  kernels for 1..N rows and the decode attention for a board
  microbenchmark (template for other kernels and targets).
- `scripts/prompt-lookup-sim.py`: estimates speculative decoding with
  prompt lookup from greedy outputs (tokens accepted per verification step,
  speedup for a given verification cost).
- `performance-optimization/scripts/ab_cli.py`: model-level A/B.
