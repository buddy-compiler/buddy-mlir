#!/bin/bash
BUDDY_BUILD_DIR := ../../build/
LLVM_BUILD_DIR := ../../llvm/build/
BUDDY_OPT := ${BUDDY_BUILD_DIR}/bin/buddy-opt
MLIR_OPT := ${LLVM_BUILD_DIR}/bin/mlir-opt
MLIR_TRANSLATE := ${LLVM_BUILD_DIR}/bin/mlir-translate
MLIR_CPU_RUNNER := ${LLVM_BUILD_DIR}/bin/mlir-cpu-runner
LLC := ${LLVM_BUILD_DIR}/bin/llc
OPT_FLAG := -O0

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.so
LIB_OMP := ${LLVM_BUILD_DIR}/lib/libomp.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.dylib
MTRIPLE := x86_64-apple-darwin
endif

linalg-spmm-f32-run:
	@${BUDDY_OPT} ./linalg-spmm-f32.mlir \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-expand-strided-metadata \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
