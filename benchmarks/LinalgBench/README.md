# LinalgBench

`LinalgBench` is a MLIR Linalg benchmark suite.
Each file in `ops/` contains one Linalg kernel function.  The Python driver
AOT-compiles each kernel and generates a small C++ host runner that uses
buddy's existing `MemRef` container from `frontend/Interfaces/buddy/Core`.
The host runner initializes tensor inputs, times each kernel loop, and executes
through:

```text
tensor linalg -> bufferization -> linalg -> VIR -> vector -> LLVM -> shared library -> C++ runtime
```

The default pipeline uses `buddy-opt` passes:

```text
-one-shot-bufferize=bufferize-function-boundaries
-buffer-deallocation-pipeline
-convert-bufferization-to-memref
-lower-linalg-to-vir
-linalg-generalize-named-ops
-lower-linalg-to-vir
-lower-vir-to-vector=vector-width=4
-cse
-convert-vector-to-scf
-lower-affine
-convert-scf-to-cf
-convert-cf-to-llvm
-convert-vector-to-llvm
-expand-strided-metadata
-lower-affine
-finalize-memref-to-llvm
-llvm-request-c-wrappers
-convert-math-to-libm
-convert-vector-to-llvm
-convert-math-to-llvm
-convert-arith-to-llvm
-convert-func-to-llvm
-reconcile-unrealized-casts
```

This is a VIR-only benchmark path.  The driver checks the IR after
`-lower-linalg-to-vir`: if no `vir.*` operations are produced, or if any
`linalg.*` operations remain, that case is reported as `no-vir` and is not
compiled further.

The two `-lower-linalg-to-vir` steps are intentional.  The first step gives
dedicated VIR patterns, such as `linalg.matmul`, a chance to match before named
ops are generalized.  The second step handles the remaining generalized
elementwise-style ops.  Convolution, pooling, and quantized generic reductions
currently lower through scalar loops with an explicit VIR output touch; the
matmul-like named ops use coarse-grained VIR vector lowering.

## Layout

```text
benchmarks/LinalgBench/
  ops/                    one tensor kernel per MLIR file
  run_linalg_bench.py     AOT build and runtime benchmark driver
```

Generated artifacts are written out of tree under
`build/benchmarks/LinalgBench/`.

The `ops/` directory groups many YAML-listed operations into fewer benchmark
files (one tensor-returning kernel per file).  It aims to cover the named
structured operations in
`llvm/mlir/include/mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.yaml`, plus
the additional non-YAML Linalg pieces listed below, **except** for the aggregate
ops that are deliberately omitted (see next paragraph).

**Not included as benchmark cases (bufferization):** `linalg.pack`,
`linalg.unpack`, and the Winograd aggregate ops (`linalg.winograd_filter_transform`,
`linalg.winograd_input_transform`, `linalg.winograd_output_transform`).  The
driver’s default one-shot bufferization does not handle these tensor aggregate
ops in this tree, so there are no `ops/*.mlir` kernels for them until that gap
is closed.

Non-YAML / extra dialect pieces exercised in the suite:

```text
generic map reduce transpose broadcast elementwise matmul contract
batch_matmul batch_reduce_matmul yield index softmax
```

`linalg.yield` and `linalg.index` are covered inside region-style benchmark
cases.

## Run

From the repository root:

```bash
python3 benchmarks/LinalgBench/run_linalg_bench.py
```

Run one case:

```bash
python3 benchmarks/LinalgBench/run_linalg_bench.py matmul
```

Tune iterations and vector width:

```bash
LINALG_BENCH_ITERS=100 LINALG_BENCH_VECTOR_WIDTH=8 \
  python3 benchmarks/LinalgBench/run_linalg_bench.py matmul conv_2d_nhwc_fhwc
```

Use scalable vectors:

```bash
python3 benchmarks/LinalgBench/run_linalg_bench.py --scalable matmul
```

The driver does not generate an MLIR `main` and does not use JIT.  It compiles
each kernel into a shared library and generates a small C++ host executable that
calls `_mlir_ciface_<kernel>` through MLIR's C interface.  Inputs and non-scalar
results use buddy's `MemRef<T, N>` container for `f32`, `f64`, `i1`, `i8`,
`i32`, and `i64`.  Rank-0 scalar tensors use MLIR's
`StridedMemRefType<T, 0>` because the current buddy `MemRef` container asserts
that rank is greater than zero.

Current limitation: because the source kernels are tensor-returning functions,
the exported C interface returns a newly allocated memref descriptor.  The AOT
runtime runner frees that result after each iteration, but the allocation/copy
cost is part of the measured kernel call.  For pure compute timing, add
memref-style exported benchmark wrappers that write into caller-owned output
buffers.

## Current VIR Status

Last checked with:

```bash
NO_COLOR=1 python3 benchmarks/LinalgBench/run_linalg_bench.py --iterations 1
```

Result:

```text
77/77 cases succeeded
```

Successful strict VIR+AOT cases:

```text
batch_matmul
batch_matmul_transpose_a
batch_matmul_transpose_b
batch_mmt4d
batch_matvec
batch_reduce_matmul
batch_vecmat
contract
conv_1d_basic
conv_1d_ncw_fcw
conv_1d_nwc_wcf
conv_2d_basic
conv_2d_nchw_fchw
conv_2d_nchw_fchw_q
conv_2d_ngchw_fgchw
conv_2d_ngchw_gfchw
conv_2d_ngchw_gfchw_q
conv_2d_nhwc_fhwc
conv_2d_nhwc_fhwc_q
conv_2d_nhwc_hwcf
conv_2d_nhwc_hwcf_q
conv_2d_nhwgc_gfhwc
conv_2d_nhwgc_gfhwc_q
conv_3d_basic
conv_3d_ncdhw_fcdhw
conv_3d_ndhwc_dhwcf
conv_3d_ndhwc_dhwcf_q
copy
core_structured_ops
depthwise_conv_1d_ncw_cw
depthwise_conv_1d_nwc_wc
depthwise_conv_1d_nwc_wcm
depthwise_conv_2d_nchw_chw
depthwise_conv_2d_nhwc_hwc
depthwise_conv_2d_nhwc_hwc_q
depthwise_conv_2d_nhwc_hwcm
depthwise_conv_2d_nhwc_hwcm_q
depthwise_conv_3d_ncdhw_cdhw
depthwise_conv_3d_ndhwc_dhwc
depthwise_conv_3d_ndhwc_dhwcm
dot
elementwise_add
fill
fill_rng_2d
generic_index
matmul
matmul_transpose_a
matmul_transpose_b
matvec
mmt4d
named_binary_f32
named_integer_select
named_unary_f32
pooling_nchw_max
pooling_nchw_sum
pooling_ncw_max
pooling_ncw_sum
pooling_ndhwc_max
pooling_ndhwc_min
pooling_ndhwc_sum
pooling_nhwc_max
pooling_nhwc_max_unsigned
pooling_nhwc_min
pooling_nhwc_min_unsigned
pooling_nhwc_sum
pooling_nwc_max
pooling_nwc_max_unsigned
pooling_nwc_min
pooling_nwc_min_unsigned
pooling_nwc_sum
quantized_batch_matmul
quantized_matmul
reduce_sum
softmax
transpose
vecmat
```

Current lowering modes:

```text
coarse VIR named lowering:
batch_matmul
batch_matmul_transpose_a
batch_matmul_transpose_b
batch_mmt4d
batch_matvec
batch_reduce_matmul
batch_vecmat
contract
dot
matmul
matmul_transpose_a
matmul_transpose_b
matvec
mmt4d
vecmat

generic VIR vector lowering:
copy
core_structured_ops
elementwise_add
fill
named_binary_f32
named_integer_select
named_unary_f32
reduce_sum
transpose

scalar loop lowering with explicit VIR output touch:
convolution/depthwise/pooling groups
conv_2d_nchw_fchw_q
conv_2d_ngchw_gfchw_q
conv_2d_nhwc_fhwc_q
conv_2d_nhwc_hwcf_q
conv_2d_nhwgc_gfhwc_q
conv_3d_ndhwc_dhwcf_q
depthwise_conv_1d_ncw_cw
depthwise_conv_1d_nwc_wc
depthwise_conv_1d_nwc_wcm
depthwise_conv_2d_nchw_chw
depthwise_conv_2d_nhwc_hwc
depthwise_conv_2d_nhwc_hwc_q
depthwise_conv_2d_nhwc_hwcm
depthwise_conv_2d_nhwc_hwcm_q
depthwise_conv_3d_ncdhw_cdhw
depthwise_conv_3d_ndhwc_dhwc
depthwise_conv_3d_ndhwc_dhwcm
quantized_batch_matmul
quantized_matmul
fill_rng_2d
generic_index
softmax
```

## Tool Paths

By default the driver uses:

```text
BUDDY_OPT              build/bin/buddy-opt
MLIR_TRANSLATE         llvm/build/bin/mlir-translate
CLANGXX                llvm/build/bin/clang++
MLIR_C_RUNNER_UTILS    llvm/build/lib/libmlir_c_runner_utils.so
```

Override them when your build tree is elsewhere:

```bash
BUDDY_OPT=/path/to/buddy-opt \
MLIR_TRANSLATE=/path/to/mlir-translate \
CLANGXX=/path/to/clang++ \
MLIR_C_RUNNER_UTILS=/path/to/libmlir_c_runner_utils.so \
python3 benchmarks/LinalgBench/run_linalg_bench.py
```

Results are written to `build/benchmarks/LinalgBench/results/linalg_bench.csv`.
Generated runtime artifacts are written to
`build/benchmarks/LinalgBench/generated/` so the lowering IR, shared library,
and host runner can be inspected or replayed manually.  Set
`LINALG_BENCH_OUT_DIR=/path/to/output` to use another out-of-tree location.
