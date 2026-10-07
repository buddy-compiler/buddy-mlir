---
name: performance-optimization
description: Measurement-driven workflow for making Buddy-MLIR faster - kernels, generated code, runtime, model builds. Use when optimizing latency or throughput, investigating a slowdown, deciding whether an optimization is worth doing, or comparing Buddy against llama.cpp or another baseline.
---

# Performance optimization

Work from measurements, one mechanism per change. The steps below are the
order; skip none of them.

## 1. Baseline

- Build the baseline from the exact commit the change will be based on, with
  the same spec, build options and run command as the candidate.
- Record: commit, spec, hardware, thread count, command, inputs (prompt
  lengths), and the output (text or a hash of it).
- Run at least twice, alternating baseline and candidate (A B A B), not all
  of A then all of B: boards drift with temperature and background load.

## 2. Upper bound before work

Estimate what the change can gain at best before writing it:

- Profile (`perf record` / `perf report --sort sym`) and get each part's
  share of the time *inside the measured window* (exclude loading).
- Compute the hardware floor of the part: bytes / bandwidth for
  bandwidth-bound work, instruction latency x count for latency-bound work.
- gain <= (current - floor) x share. If that is a few percent, say so and
  propose something else instead of implementing it.

Classify the bottleneck: DRAM bandwidth, cache / load-path bandwidth,
instruction latency (in-order cores, reductions), register spills,
synchronization / load imbalance, runtime overhead (copies, allocation,
thread start). `references/methodology.md` has the procedures.

## 3. Hypothesis, then the smallest change

State the mechanism ("the unrolled block spills; looping it removes the
spills") and the expected gain. Implement only that.

## 4. Check that the build contains the change

Before any timing, confirm the generated artifact changed: grep the
generated MLIR, or disassemble the object (`llvm-objdump -d`) and count the
instructions that should change. Stale imports and stale build directories
silently produce "no speedup" (`references/pitfalls.md`).

## 5. Correctness and numerics

Decide which class the change is in (`references/numerics-policy.md`):

- **bit-identical**: same operations in the same order. Prove it (identical
  output bytes / text hash vs the baseline) and say so in the PR.
- **numerics change**: different rounding (summation order, precision,
  quantization). Measure accuracy (perplexity), report it, and get the
  maintainer's agreement.

Run `ninja -C build check-buddy` and the tests of the touched code.

## 6. Benchmark and explain

Report baseline, candidate and the delta per input size, with the
run-to-run spread. A delta inside the spread is not a speedup. Explain why it
is faster, with the evidence (profile share, instruction counts, a
microbenchmark).

Record the result, positive or negative, in the platform's reference file
(e.g. `llm-inference-optimization/references/spacemit-k3-*.md`), including
the ideas that were measured and rejected.

## References

- `references/methodology.md`: profiling, microbenchmarks, upper-bound
  estimates, reading disassembly.
- `references/numerics-policy.md`: bit-identical vs numerics-changing
  changes and how to validate each.
- `references/pitfalls.md`: mistakes that produced wrong conclusions before.
- Platform facts: the `hardware-targets` skill.
