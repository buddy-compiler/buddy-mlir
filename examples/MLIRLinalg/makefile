#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-runner
LLC := ../../llvm/build/bin/llc
OPT_FLAG := -O0

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so
MLIR_ASYNC_RUNTIME := ../../llvm/build/lib/libmlir_async_runtime.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
MLIR_ASYNC_RUNTIME := ./../llvm/build/lib/libmlir_async_runtime.dylib
MTRIPLE := x86_64-apple-darwin
endif

linalg-conv2d-lower:
	@${MLIR_OPT} ./linalg-conv2d.mlir \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts \
		-o ./log.mlir

linalg-conv2d-translate:
	@${MLIR_OPT} ./linalg-conv2d.mlir \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-conv2d-run:
	@${MLIR_OPT} linalg-conv2d.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-conv2d-tiling-lower:
	@${MLIR_OPT} ./linalg-conv2d.mlir \
		-test-transform-dialect-interpreter \
		-o ./log.mlir

linalg-conv2d-tiling-translate:
	@${MLIR_OPT} ./linalg-conv2d.mlir \
		-test-transform-dialect-interpreter \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-conv2d-tiling-run:
	@${MLIR_OPT} linalg-conv2d.mlir ${MLIR_OPT_OPTIONS} \
		-test-transform-dialect-erase-schedule \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-conv2d_nhwc_fhwc-optimize-lower:
	@${BUDDY_OPT} linalg-conv2d_nhwc_fhwc.mlir \
		-conv-nhwc-fhwc-optimize="vec-size=16" \
	  -o ./log.mlir

linalg-conv2d_nhwc_fhwc-optimize-run:
	@${BUDDY_OPT} linalg-conv2d_nhwc_fhwc.mlir ${MLIR_OPT_OPTIONS} \
		-conv-nhwc-fhwc-optimize="vec-size=16" \
		-lower-affine -convert-scf-to-cf -convert-cf-to-llvm\
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}


linalg-conv2d_nhwc_fhwc-tile-optimize-lower:
	@${BUDDY_OPT} linalg-conv2d_nhwc_fhwc.mlir \
		-conv-nhwc-fhwc-tile-optimize="vec-size=16 tiling-height=2 tiling-width=3" \
	  -o ./log.mlir

linalg-conv2d_nhwc_fhwc-tile-optimize-run:
	@${BUDDY_OPT} linalg-conv2d_nhwc_fhwc.mlir ${MLIR_OPT_OPTIONS} \
		-conv-nhwc-fhwc-tile-optimize="vec-size=16 tiling-height=2 tiling-width=3" \
		-lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-depthwise_conv_2d_nhwc_hwc-optimize-lower:
	@${BUDDY_OPT} linalg-depthwise_conv_2d_nhwc_hwc.mlir \
		-depthwise-conv-nhwc-hwc-optimize="vec-size=16" \
	 -o ./log.mlir

linalg-depthwise_conv_2d_nhwc_hwc-optimize-run:
	@${BUDDY_OPT} linalg-depthwise_conv_2d_nhwc_hwc.mlir \
		-depthwise-conv-nhwc-hwc-optimize="vec-size=16" \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts  | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-generic-lower:
	@${MLIR_OPT} ./linalg-generic.mlir \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts \
		-o ./log.mlir

linalg-generic-translate:
	@${MLIR_OPT} ./linalg-generic.mlir \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-generic-run:
	@${MLIR_OPT} linalg-generic.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-matmul-lower:
	@${MLIR_OPT} linalg-matmul.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts \
		-o ./log.mlir

linalg-matmul-translate:
	@${MLIR_OPT} linalg-matmul.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-matmul-run:
	@${MLIR_OPT} linalg-matmul.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-matmul-optimize-lower:
	@${BUDDY_OPT} linalg-matmul.mlir ${MLIR_OPT_OPTIONS} \
		--matmul-optimize="vec-size=16 kernel-m=2 kernel-n=4" \
		-o ./log.mlir

linalg-matmul-optimize-translate:
	@${BUDDY_OPT} linalg-matmul.mlir ${MLIR_OPT_OPTIONS} \
		--matmul-optimize="vec-size=16 kernel-m=2 kernel-n=4" \
		-convert-linalg-to-loops \
		-expand-strided-metadata \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-matmul-optimize-run:
	@${BUDDY_OPT} linalg-matmul.mlir ${MLIR_OPT_OPTIONS} \
		-matmul-optimize="vec-size=16 kernel-m=2 kernel-n=4" \
		-convert-linalg-to-loops \
		-expand-strided-metadata \
		-lower-affine \
		-convert-vector-to-scf \
		-convert-vector-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-batch-matmul-optimize-run:
	@${BUDDY_OPT} linalg-batch-matmul-f32.mlir ${MLIR_OPT_OPTIONS} \
		-batchmatmul-optimize="vector-size=64" \
		-convert-linalg-to-loops \
		-expand-strided-metadata \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e buddy_batchmatmul_f32 -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-batch-matmul-lower:
	@${MLIR_OPT} linalg-batch-matmul-f32.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts \
		-o ./log.mlir

linalg-batch-matmul-translate:
	@${MLIR_OPT} linalg-batch-matmul-f32.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-batch-matmul-run:
	@${MLIR_OPT} linalg-batch-matmul-f32.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e buddy_batchmatmul_f32 -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-batch-matmul-optimize-lower:
	@${BUDDY_OPT} linalg-batch-matmul-f32.mlir ${MLIR_OPT_OPTIONS} \
		-batchmatmul-optimize="vector-size=64" \
		-o ./log.mlir

linalg-batch-matmul-tile-optimize-lower:
	@${BUDDY_OPT} linalg-batch-matmul-dync.mlir ${MLIR_OPT_OPTIONS} \
		-batchmatmul-tile-optimize="vec-size=64 kernel-m=4 kernel-n=2" \
		-o ./log.mlir

linalg-batch-matmul-tile-optimize-run:
	@${BUDDY_OPT} linalg-batch-matmul-dync.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops \
		-expand-strided-metadata \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-batch-matmul-scf-optimize-lower:
	@${BUDDY_OPT} linalg-batch-matmul-dync.mlir ${MLIR_OPT_OPTIONS} \
		-batchmatmul-scf-optimize="vector-size=64" \
		-o ./log.mlir

linalg-batch-matmul-scf-optimize-run:
	@${BUDDY_OPT} linalg-batch-matmul-dync.mlir ${MLIR_OPT_OPTIONS} \
		-batchmatmul-scf-optimize="vector-size=64" \
		-convert-linalg-to-loops \
		-expand-strided-metadata \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-batch-matmul-optimize-translate:
	@${BUDDY_OPT} linalg-batch-matmul-f32.mlir ${MLIR_OPT_OPTIONS} \
		-batchmatmul-optimize="vector-size=64" \
		-convert-linalg-to-loops \
		-expand-strided-metadata \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-batch-matmul-i8-optimize-run:
	@${BUDDY_OPT} linalg-batch-matmul-i8.mlir ${MLIR_OPT_OPTIONS} \
		-batchmatmul-optimize="vector-size=64" \
		-convert-linalg-to-loops \
		-expand-strided-metadata \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e buddy_batchmatmul_i8 -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-batch-matmul-i8-lower:
	@${MLIR_OPT} linalg-batch-matmul-i8.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts \
		-o ./log.mlir

linalg-batch-matmul-i8-translate:
	@${MLIR_OPT} linalg-batch-matmul-i8.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-batch-matmul-i8-run:
	@${MLIR_OPT} linalg-batch-matmul-i8.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e buddy_batchmatmul_i8 -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-batch-matmul-i8-optimize-lower:
	@${BUDDY_OPT} linalg-batch-matmul-i8.mlir ${MLIR_OPT_OPTIONS} \
		-batchmatmul-optimize="vector-size=64" \
		-o ./log.mlir

linalg-batch-matmul-i8-optimize-translate:
	@${BUDDY_OPT} linalg-batch-matmul-i8.mlir ${MLIR_OPT_OPTIONS} \
		-batchmatmul-optimize="vector-size=64" \
		-convert-linalg-to-loops \
		-expand-strided-metadata \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-matmul-parallized-vectorized-optmize-run:
	@${BUDDY_OPT} linalg-matmul-opt-f32.mlir ${MLIR_OPT_OPTIONS} \
		-matmul-parallel-vectorization-optimize="vector-size=128" \
		-convert-linalg-to-loops \
		-expand-strided-metadata \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e buddy_matmul_f32 -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-matmul-parallized-vectorized-optmize-lower:
	@${BUDDY_OPT} linalg-matmul-opt-f32.mlir ${MLIR_OPT_OPTIONS} \
		-matmul-parallel-vectorization-optimize="vector-size=128" \
		-o ./log.mlir

linalg-matmul-parallized-vectorized-optmize-translate:
	@${BUDDY_OPT} linalg-matmul-opt-f32.mlir ${MLIR_OPT_OPTIONS} \
		-matmul-parallel-vectorization-optimize="vector-size=128" \
		-convert-linalg-to-loops \
		-expand-strided-metadata \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-matmul-i8-parallized-vectorized-optmize-run:
	@${BUDDY_OPT} linalg-matmul-opt-i8.mlir ${MLIR_OPT_OPTIONS} \
		-matmul-parallel-vectorization-optimize="vector-size=128" \
		-convert-linalg-to-loops \
		-expand-strided-metadata \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e buddy_matmul_i8 -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-matmul-i8-parallized-vectorized-optmize-lower:
	@${BUDDY_OPT} linalg-matmul-opt-i8.mlir ${MLIR_OPT_OPTIONS} \
		-matmul-parallel-vectorization-optimize="vector-size=128" \
		-o ./log.mlir

linalg-matmul-i8-parallized-vectorized-optmize-translate:
	@${BUDDY_OPT} linalg-matmul-opt-i8.mlir ${MLIR_OPT_OPTIONS} \
		-matmul-parallel-vectorization-optimize="vector-size=128" \
		-convert-linalg-to-loops \
		-expand-strided-metadata \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-transpose-optimize-run:
	@${BUDDY_OPT} linalg-transpose-f32.mlir ${MLIR_OPT_OPTIONS} \
		-transpose-optimize="vector-size=16" \
		-lower-affine \
		-convert-vector-to-scf \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-llvm-request-c-wrappers \
		-convert-func-to-llvm \
		-lower-affine \
		-convert-arith-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e buddy_transpose_f32 -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-transpose-lower:
	@${MLIR_OPT} linalg-transpose-f32.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts \
		-o ./log.mlir

linalg-transpose-translate:
	@${MLIR_OPT} linalg-transpose-f32.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-transpose-run:
	@${MLIR_OPT} linalg-transpose-f32.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e buddy_transpose_f32 -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-transpose-optimize-lower:
	@${BUDDY_OPT} linalg-transpose-f32.mlir ${MLIR_OPT_OPTIONS} \
		-transpose-optimize="vector-size=16" \
		-o ./log.mlir

linalg-transpose-optimize-translate:
	@${BUDDY_OPT} linalg-transpose-f32.mlir ${MLIR_OPT_OPTIONS} \
		-transpose-optimize="vector-size=16" \
		-lower-affine \
		-convert-vector-to-scf \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-llvm-request-c-wrappers \
		-convert-func-to-llvm \
		-lower-affine \
		-convert-arith-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll


linalg-conv2d_nchw_fchw-lower:
	@${MLIR_OPT} ./linalg-conv2d_nchw_fchw.mlir \
		-convert-linalg-to-loops -o ./log.mlir

linalg-conv2d_nchw_fchw-translate:
	@${MLIR_OPT} ./linalg-conv2d_nchw_fchw.mlir \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-conv2d_nchw_fchw-run:
	@${MLIR_OPT} linalg-conv2d_nchw_fchw.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-conv2d_nchw_fchw-optimize-lower:
	@${BUDDY_OPT} ./linalg-conv2d_nchw_fchw.mlir \
		--conv-optimize="kernel-m=2 kernel-n=2 vec-size=16" \
		-o ./log.mlir

linalg-conv2d_nchw_fchw-optimize-translate:
	@${BUDDY_OPT} ./linalg-conv2d_nchw_fchw.mlir \
		--conv-optimize="kernel-m=2 kernel-n=3 vec-size=16" \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-conv2d_nchw_fchw-optimize-run:
	@${BUDDY_OPT} ./linalg-conv2d_nchw_fchw.mlir ${MLIR_OPT_OPTIONS} \
		--conv-optimize="kernel-m=2 kernel-n=3 vec-size=16" \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-matmul-vectorization-lower:
	@${BUDDY_OPT} linalg-matmul.mlir \
		-matmul-vectorization \
		-o log.mlir

linalg-pooling-nhwc-max-run:
	@${BUDDY_OPT} linalg-pooling-nhwc-max.mlir \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main \
		-entry-point-result=void \
	 	-shared-libs=${MLIR_RUNNER_UTILS} \
	 	-shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-pooling-nhwc-max-vectorization-lower:
	@${BUDDY_OPT} linalg-pooling-nhwc-max.mlir \
		-pooling-nhwc-max-vectorization \
		-o log.mlir

linalg-pooling-nhwc-max-vectorization-run:
	@${BUDDY_OPT} linalg-pooling-nhwc-max.mlir \
		-pooling-nhwc-max-vectorization \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main \
		-entry-point-result=void \
	 	-shared-libs=${MLIR_RUNNER_UTILS} \
	 	-shared-libs=${MLIR_C_RUNNER_UTILS}
