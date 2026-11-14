# RVV code generation pitfalls

Patterns seen in kernels that Buddy-MLIR builds with the MLIR Python
bindings (`frontend/Python/graph/transform/k3_w4.py`) and lowers through
LLVM to RVV. This file describes the mechanisms; instruction costs and
measured cases are in the platform files ("Measured cases" below).

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

Same operations in the same order: bit-identical results.

## Extract + broadcast becomes `vrgather`

`vector.extract v[u]` followed by `vector.broadcast` (e.g. a probability
scaling a value row) is folded by LLVM into a splat of lane u, lowered as
`vrgather.vi` at the destination's LMUL, which can cost several fmas at a
high LMUL. Reading the element as a scalar (from memory, or `vfmv.f.s` after a
cheap slide at a small LMUL) lets LLVM use `vfmacc.vf`.

## Reductions on in-order cores

`vector.reduction <add>` with `reassoc` lowers to `vfredusum` (unordered),
a long-latency instruction whose cost grows with LMUL; a chain of dependent
reductions is slower still. One reduction per element of a loop makes the
loop latency-bound. Changing the reduction tree (e.g. adding the LMUL-1
parts of a group first, then reducing one register) is faster but changes
the rounding: a numerics change.

## Other observations

- Strided loads (`vlse*`) can be far slower than unit-stride ones:
  transpose data in memory instead of loading columns.
- On some cores a vector load costs about the same whatever its size: then
  fewer, larger loads are better.
- Widening multiply-adds have their own LMUL constraints; check that a
  fractional-LMUL form is supported by the core before relying on it.
- On in-order cores, schedule with the core's model (`-mcpu=<core>`) and
  try `-misched-prera-direction=topdown`, which issues a block's independent
  loads early instead of next to their uses.

## Measured cases

- SpacemiT K3 A100 instruction costs, strided loads, the `vfwmacc` SIGILL:
  `hardware-targets/references/spacemit-k3-instruction-costs.md`.
- Decode attention: spills and `vrgather` removed by looping the key block,
  and the effect of top-down scheduling:
  `llm-inference-optimization/references/spacemit-k3-decode.md`.
- IME step and top-down scheduling:
  `llm-inference-optimization/references/spacemit-k3-prefill.md`.
