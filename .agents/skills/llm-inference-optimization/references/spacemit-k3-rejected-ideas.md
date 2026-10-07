# SpacemiT K3: ideas measured and not pursued

Each entry: the idea, the measurement, why it was dropped. Revisit only with
new evidence (new hardware, firmware, compiler, or a changed bottleneck).
DeepSeek-R1-1.5B `w4g32`, October 2026.

## Decode

- **IME for decode matmuls.** A one-row IME kernel reached 20.6 GB/s; the
  RVV one-row kernels reach 25 GB/s. `vmadot` computes 8 rows, so one row
  wastes 7/8 of it, and the IME weight layout is not faster to stream.
- **f16 accumulation in the decode matmuls.** ~5.7% faster; changes the
  text. Not worth a numerics change for the gain.
- **f16 KV cache.** At most ~2.5% at long contexts (KV reads are a small
  part of the bytes per token below 1k tokens).
- **Key split (flash-decoding) of decode attention.** Before #967: 253 ->
  205 us per layer at position 900 (3 heads x 2 splits), slower at short
  contexts; after #967: 146 -> 129 us. Changes the summation order. Kept as
  an option for long contexts only.
- **Smaller attention blocks (8 or 4 keys)** to cut spills: measured slower
  at every position (253 -> 276 / 283 us at position 900); looping the
  16-key block (#967) was the fix.
- **Removing the decode "fixed overhead"** (barriers and pool spinning are
  ~20% of samples): most of it is threads waiting while others saturate
  DRAM; at most ~2.4 ms per token (8%) is above the floor in total,
  realistically 1-3%.
- **Computing parts of the next token early** (decode leaves compute idle):
  every layer of token t+1 depends on token t, chosen from token t's
  logits; only speculative decoding gets around this.

## Speculative decoding (evaluated, not built)

- Verification with the RVV multi-row tiles is a loss: 4 rows cost 2.2-2.5x
  one row and 8 rows 4.3-4.8x (compute-bound), while one row is
  bandwidth-bound.
- Prompt lookup (n-gram 3..1, up to 7 drafts) on greedy outputs of 5
  realistic prompts: 1.31 tokens per step on 512-token outputs, 1.51 on
  outputs up to 1536 tokens (writing code 1.2-1.9, editing code 1.15-1.34,
  summarizing 1.4-1.6, math 1.4-1.5, open questions 1.15-1.27). A Chinese
  prompt reached 3.4 only because the model looped; excluded.
- With an 8-row IME verification kernel (estimated 1.2-1.3x a decode step,
  not measured): 1.15-1.3x decode. Needs that kernel, an IME copy of the LM
  head (+131 MB), a verification graph and runtime support; output differs
  from plain decode (fp16 IME). Next step if resumed: build and time the
  8-row IME kernel; drop the idea if verification costs 1.5x or more.

## Prefill

- **DMA prefetch of IME weights into TCM.** With activations already in TCM
  (#963) the step takes 62.8 ns per group; with the weights in TCM too a
  microbenchmark gave 56.9 ns: ~3% of prefill at best, before the cost of
  ~6 us DMA requests, double buffering and synchronization. (An earlier
  15-25% estimate reused the gain #963 had already taken.)
