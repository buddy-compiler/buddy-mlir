#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
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
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts \
		-o ./log.mlir

linalg-conv2d-translate:
	@${MLIR_OPT} ./linalg-conv2d.mlir \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-conv2d-run:
	@${MLIR_OPT} linalg-conv2d.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-conv2d-tiling-lower:
	@${MLIR_OPT} ./linalg-conv2d.mlir \
		-test-transform-dialect-interpreter \
		-o ./log.mlir

linalg-conv2d-tiling-translate:
	@${MLIR_OPT} ./linalg-conv2d.mlir \
		-test-transform-dialect-interpreter \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-conv2d-tiling-run:
	@${MLIR_OPT} linalg-conv2d.mlir ${MLIR_OPT_OPTIONS} \
		-test-transform-dialect-interpreter \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-generic-lower:
	@${MLIR_OPT} ./linalg-generic.mlir \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts \
		-o ./log.mlir

linalg-generic-translate:
	@${MLIR_OPT} ./linalg-generic.mlir \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-generic-run:
	@${MLIR_OPT} linalg-generic.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-matmul-lower:
	@${MLIR_OPT} linalg-matmul.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts \
		-o ./log.mlir

linalg-matmul-translate:
	@${MLIR_OPT} linalg-matmul.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-matmul-run:
	@${MLIR_OPT} linalg-matmul.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-matmul-optimize-lower:
	@${BUDDY_OPT} linalg-matmul.mlir ${MLIR_OPT_OPTIONS} \
		--matmul-optimize="vec-size=16 kernel-m=2 kernel-n=4" \
		-o ./log.mlir

linalg-matmul-optimize-translate:
	@${BUDDY_OPT} linalg-matmul.mlir ${MLIR_OPT_OPTIONS} \
		--matmul-optimize="vec-size=16 kernel-m=2 kernel-n=4" -convert-linalg-to-loops \
		-lower-affine -convert-scf-to-cf -convert-vector-to-llvm \
		-convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-matmul-optimize-run:
	@${BUDDY_OPT} linalg-matmul.mlir ${MLIR_OPT_OPTIONS} \
		--matmul-optimize="vec-size=16 kernel-m=2 kernel-n=4" -convert-linalg-to-loops \
		-lower-affine -convert-scf-to-cf -convert-vector-to-llvm \
		-convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-conv2d_nchw_fchw-lower:
	@${MLIR_OPT} ./linalg-conv2d_nchw_fchw.mlir \
		-convert-linalg-to-loops -o ./log.mlir

linalg-conv2d_nchw_fchw-translate:
	@${MLIR_OPT} ./linalg-conv2d_nchw_fchw.mlir \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-conv2d_nchw_fchw-run:
	@${MLIR_OPT} linalg-conv2d_nchw_fchw.mlir ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

linalg-conv2d_nchw_fchw-optimize-lower:
	@${BUDDY_OPT} ./linalg-conv2d_nchw_fchw.mlir \
		--conv-optimize="kernel-m=2 kernel-n=2 vec-size=16" \
		-o ./log.mlir

linalg-conv2d_nchw_fchw-optimize-translate:
	@${BUDDY_OPT} ./linalg-conv2d_nchw_fchw.mlir \
		--conv-optimize="kernel-m=2 kernel-n=3 vec-size=16" \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

linalg-conv2d_nchw_fchw-optimize-run:
	@${BUDDY_OPT} ./linalg-conv2d_nchw_fchw.mlir ${MLIR_OPT_OPTIONS} \
		--conv-optimize="kernel-m=2 kernel-n=3 vec-size=16" \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

