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

openacc-serial-lower:
	@${MLIR_OPT} openacc-serial.mlir | \
	${MLIR_OPT} -convert-openacc-to-scf -convert-scf-to-cf -convert-func-to-llvm -convert-cf-to-llvm -o log.mlir

openacc-bound-lower:
	@${MLIR_OPT} openacc-bound.mlir | \
	${MLIR_OPT} -convert-openacc-to-scf -convert-scf-to-cf -convert-func-to-llvm -convert-cf-to-llvm -o log.mlir

openacc-update-lower:
	@${MLIR_OPT} openacc-update.mlir | \
	${MLIR_OPT} -convert-openacc-to-scf -convert-scf-to-cf -convert-func-to-llvm -convert-cf-to-llvm -o log.mlir

openacc-data-lower:
	@${MLIR_OPT} openacc-data.mlir | \
	${MLIR_OPT} -convert-openacc-to-scf -convert-scf-to-cf -convert-func-to-llvm -convert-cf-to-llvm -o log.mlir

openacc-loop-lower:
	@${MLIR_OPT} openacc-loop.mlir | \
	${MLIR_OPT} -convert-openacc-to-scf -convert-scf-to-cf -convert-func-to-llvm -convert-cf-to-llvm -o log.mlir



