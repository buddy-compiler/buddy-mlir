# Prefill

## Budget a prefill

Profile one prefill of a long prompt (several hundred tokens) and split the
time into: matmul inner loops (the matrix-unit or vector microkernel),
tile overhead (packing, accumulator traffic, copies into on-chip memory),
synchronization, attention, everything else. The microkernel's cost per
unit of work (one call of its inner loop body) times the work count should
match its profile share; compare it with the same microkernel
alone on one core to see whether contention or the kernel itself is the
limit.

## Levers

In the order they paid off on the one platform measured so far (results:
`spacemit-k3-prefill.md`):

1. Run the matmul tiles on the matrix unit instead of vector tiles.
2. Remove copies around external kernel calls (bufferization copies of
   every argument) and prefault the model's memory arena at load.
3. Feed the matrix unit from fast on-chip memory (TCM) when cached loads are
   contended across cores.
4. Attention on the matrix unit for long prompts.
5. Skip work whose result is unused (the LM head of all chunks but the
   last; only the last row's logits are needed).

## Chunked prefill

The prompt runs in chunks of a fixed row count (set by the matrix tiles;
see the platform file), each like a decode step of many rows over the KV
cache (`docs/ChunkedPrefill.md`). The last chunk is padded: prompts just above a
multiple of the chunk size pay for a whole chunk.

## What does not pay (check before retrying)

- Prefetching weights into on-chip memory by DMA when the activations
  already come from it: once the activations no longer contend, the weight
  path may be a small part of the microkernel's time; bound it first
  (measured case: `spacemit-k3-rejected-ideas.md`).
