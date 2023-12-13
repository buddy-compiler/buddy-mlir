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
MLIR_CUDA_RUNTIME := ../../llvm/build/lib/libmlir_cuda_runtime.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
MLIR_ASYNC_RUNTIME := ./../llvm/build/lib/libmlir_async_runtime.dylib
MTRIPLE := x86_64-apple-darwin
endif

openacc-update-lower:
	@${MLIR_OPT} openacc-update.mlir -convert-openacc-to-scf | \
	${MLIR_OPT} -convert-scf-to-cf |\
	${MLIR_OPT}	-convert-func-to-llvm -convert-cf-to-llvm -o log.mlir

openacc-update-translate:
	@${MLIR_OPT} openacc-update.mlir -convert-openacc-to-scf | \
	${MLIR_OPT} -convert-scf-to-cf |\
	${MLIR_OPT}	-convert-func-to-llvm -convert-cf-to-llvm | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

openacc-update-run:
	@${MLIR_OPT} openacc-update.mlir -convert-openacc-to-scf | \
	${MLIR_OPT} -convert-scf-to-cf |\
	${MLIR_OPT}	-convert-cf-to-llvm -convert-func-to-llvm| \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}