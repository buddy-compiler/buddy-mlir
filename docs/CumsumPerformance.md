# Cumsum CPU traversal

For static rank-two f32/i64 inputs scanned along axis zero with at least
64 columns and a nonempty first axis, visit consecutive columns together.
Each column still starts with zero plus its first value and accumulates in
increasing row order. The previous output row supplies the accumulator;
there is no scratch allocation or reassociation. Other shapes, axes and
output types retain the existing implementation.

This targets strided loads and stores in the original column-at-a-time
loop. It adds a load of the previous output row, which is usually cache
resident. The initial addition is retained for signed-zero semantics.

## Measurement

Compare the same benchmark against baseline commit 60f354a and the candidate,
using the same LLVM build, CPU and input seed. Run baseline, candidate,
baseline, candidate. Native timing includes output allocation and release;
export, compilation and Python tensor conversion are excluded.

```sh
python scripts/benchmark_cumsum.py --rows 1024 --width 1024 --dtype f32 --dim 0
python scripts/benchmark_cumsum.py --rows 1024 --width 1024 --dtype i64 --dim 0
```

On an i9-14900HX CPU container with four visible CPUs, PyTorch 2.10.0+cpu,
one Torch thread, and LLVM 3845a205, two rounds of median native latency
for 1024-by-1024 inputs were 5.249/5.031 ms versus 0.151/0.158 ms (f32),
and 4.205/3.832 ms versus 0.307/0.359 ms (i64). Each round used five warmups
and five batches of ten calls. These are CPU microbenchmarks, not K3 or
model-level results. Absolute timings fluctuate with the host.

Check threshold neighbors (63/64/65 columns), short and empty axes, axis
one and higher ranks as well. The unchanged path must retain identical
IR; timing fluctuations there are not evidence of an algorithm change.
The unchanged f32 1024-by-1024 axis-one path measured 0.437/0.409 ms
versus 0.418/0.397 ms. Treat these JIT timings as subject to allocation
and code-placement effects, not a speedup guarantee.

The JIT test compares against sequential additions including the initial
zero. Random floating-point sums need not match a framework implementation
that uses a different reduction order. Framework-reference tests compare
NaN positions, since NaN signs and payloads can differ, and compare all
remaining values bit for bit (including signed zeros). Experimental baseline comparisons
also cover NaNs, infinities, signed zeros, cancellation and integer overflow.

```sh
ninja -C build check-buddy
ninja -C build check-buddy-server
```
