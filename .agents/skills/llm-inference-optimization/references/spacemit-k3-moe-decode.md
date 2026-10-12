# SpacemiT K3: decode of Qwen3-VL-30B-A3B, text decoder (`w4g32`)

Spec `models/qwen3_vl_moe/specs/k3_w4g32.json` (chunk 64, IME prefill for
the dense layers, RVV MoE kernels), 8 A100 cores, `buddy-cli` greedy;
kernels in `frontend/Python/graph/transform/k3_w4.py`. Measured October
2026; compare only runs made together (~2% run to run).

## Where a decode step goes (short context)

`BUDDY_THREAD_POOL_PROFILE=1` (docs/ModelThreadPool.md), difference of a
4-token and a 196-token run. Before the changes below: 80.8 ms per token.

| Part | ms / token |
| --- | --- |
| MoE gate / up (8 experts, 14.2 MB per layer, 24.4 GB/s) | 28.0 |
| MoE down (7.1 MB per layer, 25.6 GB/s) | 13.3 |
| q / k / v with fused RMSNorm (26.8 GB/s) | 10.5 |
| o projection (26.5 GB/s) | 8.5 |
| LM head (175 MB, 26.2 GB/s) | 6.7 |
| Attention | 3.4 |
| MoE routing logits, h quantization | 3.2 |
| Serial gaps: one thread working, seven waiting | 7.2 |
| Barrier waits (load imbalance), wall equivalent | ~1.9 |

The kernels themselves stream at 24-27 GB/s; the gap to llama.cpp's
effective bandwidth was the serial time between parallel regions (337
regions per token). perf on the calling thread showed where: the graph's
glue ops (`subgraph0_decode`, scalar loops: the decode subgraph pipeline
neither vectorizes nor parallelizes them) and the serial prologues of the
kernels (activation quantization, top-k, RoPE cos / sin).

## Changes and results

| decode, tok/s | short prompt | after 458 tokens | after 900 tokens |
| --- | --- | --- | --- |
| before | 12.53 | 11.47 | 10.53 |
| RMSNorm before the MoE block fused into the MoE kernel | 12.90 | 11.84 | 10.74 |
| + residual adds fused into the o / MoE kernels, q_norm per attention item | 13.05 | 12.03 | 10.81 |
| llama.cpp, SpacemiT Q4_1 GGUF | 11.77 | 10.94 | 10.26 |

Serial time per token: 7.2 -> 5.3 ms (RMSNorm fusion) -> 5.1 ms (q_norm in
the attention items) -> lower with the residual fusion. Prefill also gained
2-3% (9.30 -> 9.11 s at 458 tokens).

- The RMSNorm fusion changes the summation order of the norm (as the fused
  norms of q / k / v and the LM head already did): perplexity, wikitext-2
  test, 512 x 16: 7.9814 before, 7.9825 after.
- The residual fusion and the q_norm move are bit-identical (same text).

## Remaining (short context, after the changes)

- Serial gaps ~5 ms per token: between tokens 0.86 ms (logits copy,
  sampling, the first glue ops), per layer ~20 us before q / k / v (RMSNorm
  and quantization of x), ~19 us before attention (cos / sin, k norm and
  RoPE, KV writes), ~16 us before the o projection (quantization of 4096
  activations), ~11 us before gate / up (routing top-k, quantization of x).
- MoE gate / up at 24.4 GB/s against 26-27 GB/s for the other kernels:
  ~2 ms per token if it matched them (see the rejected ideas for what did
  not help).
- Barrier waits ~1.9 ms per token, mostly in gate / up: one expert per
  thread, ~16 us of imbalance per call.
