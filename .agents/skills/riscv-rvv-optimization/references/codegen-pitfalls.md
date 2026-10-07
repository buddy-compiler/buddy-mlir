# RVV code generation pitfalls

Patterns seen in kernels that Buddy-MLIR builds with the MLIR Python
bindings (`frontend/Python/graph/transform/k3_w4.py`) and lowers through
LLVM to RVV. Costs quoted are from the SpacemiT K3 A100
(`hardware-targets/references/spacemit-k3-instruction-costs.md`).

## Reading the code

```bash
llvm-objdump -d --mattr=+v,+zfh,+zvfh k.o > k.asm
# instruction histogram of one function (the parallel body of a kernel)
awk '/<kernel..omp_par>:/{p=1;next} p&&/>:$/&&!/Lpcrel/{p=0} p' k.asm \
  | awk '{print $3}' | sort | uniq -c | sort -rn | head -30
```

Signs of trouble: `vs<n>r.v` / `vl<n>r.v` with an `sp`-relative address
(whole-register spills and reloads); `vrgather`; `vslideup` / `vslidedown`
at high LMUL; many `vsetvli`.

## Unrolled loops that spill

A Python loop that emits the body of a block N times (`for u in range(B)`)
gives LLVM one long basic block. With LMUL-4 vectors the 8 register groups
fill quickly (accumulators per head, query vectors, loaded rows), and the
pre-RA scheduler may hoist loads, so LLVM spills whole groups (a store and a
reload of 512 bytes each).

Fix: emit an `scf.for` over the block instead of unrolling. Values that
cross from one phase to the next (scores, probabilities) go through a small
stack buffer (`memref.alloca`, aligned like the other buffers, 128 bytes):

- write each scalar score with `memref.store`, then read the block as one
  vector with `vector.load`;
- store the probability vector once and read each element with
  `memref.load` as a scalar.

Same operations in the same order: bit-identical results. Decode attention:
46 spill stores + 55 reloads per 16-key block removed, 253 -> 146 us per
layer at position 900.

## Extract + broadcast becomes `vrgather`

`vector.extract v[u]` followed by `vector.broadcast` (e.g. a probability
scaling a value row) is folded by LLVM into a splat of lane u, lowered as
`vrgather.vi` at the destination's LMUL: 17.8 ns at LMUL 4, four times an
fma. Reading the element as a scalar (from memory, or `vfmv.f.s` after a
cheap slide at a small LMUL) lets LLVM use `vfmacc.vf`.

## Reductions on in-order cores

`vector.reduction <add>` with `reassoc` lowers to `vfredusum` (unordered,
~28 ns at LMUL 4; a dependent chain ~57 ns per reduction). One reduction per
element of a loop makes the loop latency-bound; changing the reduction tree
(e.g. adding the four LMUL-1 parts first, then an 8 ns LMUL-1 reduction) is
faster but changes the rounding: a numerics change.

## Other observations

- Strided loads (`vlse32`) are very slow (~300 ns on the K3): transpose data
  in memory instead of loading columns.
- Vector loads cost about the same whatever their size up to LMUL 8: one
  1 KiB load beats several small ones.
- Widening multiply-adds have their own LMUL limits; a `vfwmacc` at a
  fractional LMUL raised SIGILL on the K3 A100 in a hand-written
  microbenchmark.
- `-misched-prera-direction=topdown` with `-mcpu=spacemit-a100` issues a
  block's independent loads early on the in-order A100 (IME step 75 -> 59 ns
  per group; decode attention 332 -> 284 us).
