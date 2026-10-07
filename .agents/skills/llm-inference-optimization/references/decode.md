# Decode

## Budget a decode step

1. Weight bytes per token: every Linear layer plus the LM head (often 10-15%
   of a small model's weights: DeepSeek-R1-1.5B's int4 head is 131 of
   868 MB). Floor = bytes / measured read bandwidth.
2. Attention: grows linearly with the position (KV cache reads plus a
   reduction per key and head). Measure at short and long contexts.
3. Everything else: norms, RoPE, sampling, per-token runtime work, idle
   threads between parallel regions.

Compare the per-token time with the floor before choosing work. When the
matmuls already run near the bandwidth, the remaining levers are attention
(long contexts), fewer bytes per token (formats: numerics change), or more
tokens per weight read (speculative decoding, batching).

## Matmul kernels (one row)

- Bandwidth-bound: one pass over the weights, all cores streaming, enough
  columns per thread that the tiles balance. Small matrices (q/k/v, o) reach
  a lower bandwidth than large ones: start-up and imbalance dominate.
- Idle threads at a barrier do not cost time if the busy ones already
  saturate DRAM.

## Attention (one query row)

- Several heads sharing a KV head (GQA) in one work item load each key and
  value row once for all of them, and leave threads for parallelism; choose
  the fewest heads per item that keep the item count at or below the thread
  count.
- Per key, the score is a reduction: on in-order cores its latency dominates
  (see the target's instruction costs). Keep the loop body small: an
  unrolled block of keys can spill vector registers and turn per-key
  broadcasts into expensive permutes; looping keys through a small stack
  buffer kept the same operations and order and was 1.7x faster on the K3.
- Splitting one head's keys across threads (flash-decoding) adds
  parallelism but changes the summation order (numerics) and costs a merge;
  it pays only at long contexts.

## Speculative decoding

Decode leaves compute idle, so verifying k draft tokens in one weight pass
can produce several tokens per pass. It pays only if verifying k + 1 rows
costs little more than one row:

- Measure the multi-row kernels: compute-bound vector tiles can cost several
  times a one-row step (4 rows 2.3x and 8 rows 4.4x on the K3 RVV tiles),
  which makes speculation a loss. A matrix unit that computes 8 rows anyway
  is the right tool for verification.
- Estimate acceptance before building: `scripts/prompt-lookup-sim.py` on
  greedy outputs of realistic prompts. Exclude degenerate repetition loops
  (they inflate acceptance). Long reasoning outputs accept more than short
  ones.
- Verification on a different kernel (e.g. matrix engine fp16 vs vector
  f32) makes the output differ from plain decode: a numerics change.
- Needs: a multi-row decode graph with logits for every row, position
  rollback (attention reads only keys up to the row's position, so rejected
  KV entries are overwritten later), multi-token emission in the generation
  loop, and rejection sampling for non-greedy sampling.
