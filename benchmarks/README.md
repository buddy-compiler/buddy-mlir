# Benchmarks

## LinalgBench

`benchmarks/LinalgBench` contains tensor-based MLIR Linalg benchmark kernels and
a Python AOT driver that executes them through buddy-mlir's VIR route:

```text
tensor linalg -> bufferization -> linalg -> vir -> vector -> LLVM
```

This is a strict VIR-only path: cases that do not lower to `vir.*`, or still
contain residual `linalg.*` after `lower-linalg-to-vir`, fail with `no-vir`.
The `ops/` suite is intended to cover the full Linalg dialect surface: all
named structured ops from `LinalgNamedStructuredOps.yaml` plus the non-YAML
structured, aggregate, relayout, and region ops.

Run all cases:

```bash
python3 benchmarks/LinalgBench/run_linalg_bench.py
```

Run selected cases:

```bash
python3 benchmarks/LinalgBench/run_linalg_bench.py matmul pooling_nhwc_max
```

The driver compiles kernels AOT into shared libraries and calls
`_mlir_ciface_<kernel>` from generated C++ host executables that use buddy's
existing `MemRef` container.  It uses `build/bin/buddy-opt`,
`llvm/build/bin/mlir-translate`, `llvm/build/bin/clang++`, and MLIR runtime
libraries under `llvm/build/lib` by default.  See
`benchmarks/LinalgBench/README.md` for environment variables, the full lowering
pipeline, generated files under `build/benchmarks/LinalgBench/`, and CSV output.

## ModelBench

`benchmarks/ModelBench` contains end-to-end model benchmarks.  The initial case
is an F32 DeepSeek-R1 benchmark that imports the model through
`examples/BuddyDeepSeekR1`, builds prefill and decode entry points, and times
one fake-input execution of each phase.

See `benchmarks/ModelBench/README.md` for details.
