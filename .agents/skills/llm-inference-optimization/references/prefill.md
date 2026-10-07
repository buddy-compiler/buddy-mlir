# Prefill

## Budget a prefill

Profile one prefill of a long prompt (several hundred tokens) and split the
time into: matmul inner loops (the matrix-unit or vector microkernel),
tile overhead (packing, accumulator traffic, copies into on-chip memory),
synchronization, attention, everything else. The microkernel's cost per
unit of work (e.g. ns per 8-column x 32-row x 32-K group) times the work
count should match its profile share; compare it with the same microkernel
alone on one core to see whether contention or the kernel itself is the
limit.

## Levers, in the order they paid off on the K3

1. Run the matmul tiles on the matrix unit instead of vector tiles (3.4-3.7x
   on the K3).
2. Remove copies around external kernel calls (bufferization copies of
   every argument) and prefault the model's memory arena at load.
3. Feed the matrix unit from fast on-chip memory (TCM) when cached loads are
   contended across cores.
4. Attention on the matrix unit for long prompts.
5. Skip work whose result is unused (the LM head of all chunks but the
   last; only the last row's logits are needed).

## Chunked prefill

The prompt runs in chunks of a fixed row count (64 for the K3 IME tiles),
each like a decode step of many rows over the KV cache
(`docs/ChunkedPrefill.md`). The last chunk is padded: prompts just above a
multiple of the chunk size pay for a whole chunk.

## What does not pay (check before retrying)

- Prefetching weights into on-chip memory by DMA when the activations
  already come from it: on the K3 the remaining gain was ~3% of prefill
  (`spacemit-k3-rejected-ideas.md`).
