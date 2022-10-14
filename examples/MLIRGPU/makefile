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

gpu-all-reduce-and-lower:
	@${MLIR_OPT} gpu-all-reduce-and.mlir -gpu-kernel-outlining \
		-pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)' \
		-gpu-to-llvm -o log.mlir

gpu-all-reduce-and-run:
	@${MLIR_OPT} gpu-all-reduce-and.mlir -gpu-kernel-outlining \
		-pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)' \
		-gpu-to-llvm | \
	${MLIR_CPU_RUNNER} -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

gpu-launch-func-lower:
	@${MLIR_OPT} gpu-launch-func.mlir -gpu-kernel-outlining \
		-pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)' \
		-gpu-async-region -gpu-to-llvm -o log.mlir

gpu-launch-func-run:
	@${MLIR_OPT} gpu-launch-func.mlir -gpu-kernel-outlining \
		-pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)' \
		-gpu-async-region -gpu-to-llvm | \
	${MLIR_CPU_RUNNER} -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

async-execute-lower:
	@${MLIR_OPT} async-execute.mlir -gpu-kernel-outlining \
		-pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)' \
		-gpu-async-region -gpu-to-llvm -o log.mlir

async-execute-run:
	@${MLIR_OPT} async-execute.mlir -gpu-kernel-outlining \
		-pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)' \
		-gpu-async-region -gpu-to-llvm -async-to-async-runtime -async-runtime-ref-counting \
    -convert-async-to-llvm -convert-func-to-llvm | \
	${MLIR_CPU_RUNNER} -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME} \
	-shared-libs=${MLIR_ASYNC_RUNTIME} ${OPT_FLAG}

gpu-mma-lower:
	@${MLIR_OPT} gpu-mma.mlir -gpu-kernel-outlining \
		-pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin{chip=sm_70})' \
		-convert-scf-to-cf -gpu-to-llvm -async-to-async-runtime -async-runtime-ref-counting \
    -convert-async-to-llvm -convert-func-to-llvm -o log.mlir
		
gpu-mma-run:
	@${MLIR_OPT} gpu-mma.mlir -gpu-kernel-outlining \
		-pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin{chip=sm_70})' \
		-convert-scf-to-cf -gpu-to-llvm -async-to-async-runtime -async-runtime-ref-counting \
    -convert-async-to-llvm -convert-func-to-llvm | \
	${MLIR_CPU_RUNNER} -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME} \
	-shared-libs=${MLIR_ASYNC_RUNTIME} 

vector-to-gpu-lower:
	@${MLIR_OPT} vector-to-gpu.mlir -pass-pipeline="gpu.module(strip-debuginfo,convert-vector-to-gpu, \
	convert-gpu-to-nvvm,gpu-to-cubin{chip=sm_70})" -convert-scf-to-cf -gpu-to-llvm \
	-canonicalize -o log.mlir 

vector-to-gpu-run:
	@${MLIR_OPT} vector-to-gpu.mlir -pass-pipeline="gpu.module(strip-debuginfo,convert-vector-to-gpu, \
	convert-gpu-to-nvvm,gpu-to-cubin{chip=sm_70})" -convert-scf-to-cf -gpu-to-llvm \
	-canonicalize | \
	${MLIR_CPU_RUNNER} -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME} \
	-shared-libs=${MLIR_ASYNC_RUNTIME}

