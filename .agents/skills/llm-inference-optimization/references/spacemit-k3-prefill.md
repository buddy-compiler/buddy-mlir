# SpacemiT K3: prefill of DeepSeek-R1-Distill-Qwen-1.5B (`w4g32`)

Spec `models/deepseek_r1/specs/k3_w4g32.json` (`prefill_chunk` 64,
`prefill_ime`), 8 A100 cores. Measured October 2026. Details of the kernels:
`docs/K3DeepSeekR1.md`.

## History (seconds)

| Change | 64 tokens | 458 tokens | 900 tokens |
| --- | --- | --- | --- |
| RVV tiles | 2.40 | 9.6 | 17.7 |
| IME tiles (#960) | 0.70 | 2.58 | 5.03 |
| IME prefill attention (#961) | 0.69 | 2.30 | 4.28 |
| No bufferization copies of kernel arguments, prefaulted arena (#962) | 0.42 | 1.60 | 3.09 |
| IME tiles read activations from the pair's TCM (#963) | 0.34 | 1.27 | 2.48 |
| LM head only for the last chunk (#964) | | ~1.24 | |
| llama.cpp 0.1.9 Q4_0 (`llama-bench -p`) | | 1.96 | |

The IME tiles compute products in fp16 and use f16 activation scales: the
text departs from the RVV path's after 100-330 characters (numerics change,
accepted; largest kernel error 3e-4 of the largest output).

## Where a prefill goes now (458 tokens)

| Part | Share |
| --- | --- |
| `k3_ime_hp_step` (IME inner loop) | ~55% |
| IME tile overhead (accumulators, write-back) and copying activations to TCM | ~14% |
| Pool synchronization (`__kmpc_barrier`, `worker`) | ~11% |
| Attention (IME) | ~9% |
| Rest | ~10% |

The step runs ~82M groups (8 columns x 32 rows x 32 K) in 5.14 core-seconds:
62.8 ns per group, close to its single-core 59 ns. It is latency-bound on
the in-order core (three ~10 ns loads, unpack, 8 `vmadot.hp`, 4 widening
fmas), not bandwidth-bound.

## Open options (not done)

- One step for 2 or 4 column blocks: A (1 KiB) and S (512 B) loaded once for
  all of them, 2 loads per block-group instead of 3. Estimated 5-6% of
  prefill; bit-identical. Microbenchmark first.
- Accumulator traffic between K chunks of the k 8960 projection and tile
  balance (24 tiles on 8 threads): a few percent.
