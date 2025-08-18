#!/bin/bash
BUDDY_BUILD_DIR := ../../build/
LLVM_BUILD_DIR := ../../llvm/build/
BUDDY_OPT := ${BUDDY_BUILD_DIR}/bin/buddy-opt
MLIR_OPT := ${LLVM_BUILD_DIR}/bin/mlir-opt
MLIR_TRANSLATE := ${LLVM_BUILD_DIR}/bin/mlir-translate
MLIR_CPU_RUNNER := ${LLVM_BUILD_DIR}/bin/mlir-runner
LLC := ${LLVM_BUILD_DIR}/bin/llc
OPT_FLAG := -O3

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.so
MLIR_ASYNC_RUNTIME := ${LLVM_BUILD_DIR}/lib/libmlir_async_runtime.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.dylib
MLIR_ASYNC_RUNTIME := ${LLVM_BUILD_DIR}/lib/libmlir_async_runtime.dylib
MTRIPLE := x86_64-apple-darwin
endif

buddy-mlp-fusion:
	@${BUDDY_OPT} ./fake-mlp.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |\
	${BUDDY_OPT} \
		-linear-silu-fusion \
		-linear-silu-lower \
		> ./fused.mlir

buddy-mlp-fused-run:
	@${BUDDY_OPT} ./fused.mlir \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-buffer-deallocation \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-convert-vector-to-llvm \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
		
buddy-mlp-run:
	@${BUDDY_OPT} ./fake-mlp.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" | \
	${BUDDY_OPT} \
		-linear-silu-fusion \
		-linear-silu-lower \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-buffer-deallocation \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-convert-vector-to-llvm \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}