# SpacemiT K3: decode of DeepSeek-R1-Distill-Qwen-1.5B (`w4g32`)

Spec `models/deepseek_r1/specs/k3_w4g32.json`, 8 A100 cores, `buddy-cli`
greedy; kernels in `frontend/Python/graph/transform/k3_w4.py`. Measured
October 2026. Board-to-board and day-to-day variation is ~2%: compare only
runs made together.

## Where a decode step goes (short context)

| Part | ms / token |
| --- | --- |
| Linear layers (q/k/v, o, gate/up, down of 28 layers) | ~29.7 |
| LM head (151936 x 1536 int4, 131 MB) | ~5.2 |
| Attention | ~0.5 (position 63) to ~4.1 (position 900) |
| Rest (other ops, parallel-region start/stop, sampling, session) | ~1 |

Weights: 868 MB per token; at ~26 GB/s the floor is 33.5 ms (29.8 tok/s).
gate/up and down reach 24-25.7 GB/s; q/k/v and o only 20-23 GB/s (few
column tiles per thread).

`__kmpc_barrier` + pool `worker` are ~20% of perf samples, mostly not lost:
e.g. down (12 column tiles) runs 2 tiles on 4 threads and 1 on 4, but 4
cores already read 24 GB/s.

## Results

| decode, tok/s | short prompt | after 458 tokens | after 900 tokens |
| --- | --- | --- | --- |
| before #966 | 26.5 | 23.5 | 21.2 |
| #966: A100 scheduling, 2 heads per attention item | 26.3-27.0 | 24.7-25.2 | 22.7-23.3 |
| #967: attention keys looped, no spills | 27.6 | 26.5 | 25.1 |
| llama.cpp 0.1.9 Q4_0, Q4_0 LM head | 26.6 | 24.8 | 23.3 |

All three are bit-identical to their predecessors (same text).

## Decode attention (12 heads, 2 KV heads, d 128)

Per layer at position 900: 332 us (one head per item, generic scheduling)
-> 284 us (`kernels_a100` scheduling) -> 261 us (2 heads of a KV group per
item: 6 items on 8 threads) -> 146 us (#967). #967 found the time in the
disassembly: the unrolled block of 16 keys spilled 46 + 55 LMUL-4 register
groups per block and broadcast each probability with `vrgather.vi`
(17.8 ns); the keys now run as loops through a `[heads][16]` stack buffer
(128-byte aligned), the probabilities read back as scalars for
`vfmacc.vf`.

Per head and key the cost is now dominated by the `vfredusum` (~28 ns).

## Open options (numerics change, not done)

- Fold the 4 registers of a product to one before the reduction (m4 -> m1
  adds, then an 8 ns m1 reduction): ~30% less attention, ~3% decode at 900
  tokens.
- Key split (flash-decoding) on top of #967: 3 heads per item x 2 key
  halves = 8 items, 146 -> 129 us at position 900 but slower at positions 63
  and 320.
- f16 KV cache: halves KV reads; a few percent at long contexts only.
- Speculative decoding: see `spacemit-k3-rejected-ideas.md` (evaluated,
  needs an 8-row IME verification kernel first).
