#!/bin/bash
BUDDY_BUILD_DIR := ../../build/
LLVM_BUILD_DIR := ../../llvm/build/
BUDDY_OPT := ${BUDDY_BUILD_DIR}/bin/buddy-opt
MLIR_OPT := ${LLVM_BUILD_DIR}/bin/mlir-opt
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

vir-setvl-lower:
	${BUDDY_OPT} vir-setvl.mlir -o log.mlir

vir-dynamic-vec-type-lower:
	${BUDDY_OPT} vir-dynamic-vec-type.mlir \
	--allow-unregistered-dialect \
	-o log.mlir

vir-memory-lower:
	${BUDDY_OPT} vir-memory.mlir \
	-lower-vir-to-vector \
	-cse \
	-o log.mlir

vir-memory-run:
	${BUDDY_OPT} vir-memory.mlir \
		-lower-vir-to-vector \
		-cse \
		-convert-vector-to-scf \
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

vir-basic-lower:
	${BUDDY_OPT} vir-basic.mlir \
	-o log.mlir

vir-arith-lower:
	${BUDDY_OPT} vir-arith.mlir \
	-o log.mlir
