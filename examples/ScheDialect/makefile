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

.SECONDEXPANSION:
all-run: $$(run-targets)

sche-scf-for-lower:
	@${BUDDY_OPT} ./sche-scf-for.mlir -device-schedule | \
	${BUDDY_OPT} -lower-sche | \
	${MLIR_OPT} -gpu-kernel-outlining | \
	${MLIR_OPT} --pass-pipeline="builtin.module(func.func(tosa-to-linalg),empty-tensor-to-alloc-tensor,arith-bufferize, func.func(linalg-bufferize), func.func(tensor-bufferize), func-bufferize,func.func(buffer-deallocation, convert-linalg-to-loops) , convert-scf-to-cf, expand-strided-metadata, memref-expand)" | \
	${MLIR_OPT} -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin))' | \
	${MLIR_OPT} -gpu-async-region -gpu-to-llvm -o log.mlir

run-targets += sche-scf-for-run
sche-scf-for-run:
	@${BUDDY_OPT} ./sche-scf-for.mlir -device-schedule | \
	${BUDDY_OPT} -lower-sche | \
	${MLIR_OPT} -gpu-kernel-outlining | \
	${MLIR_OPT} --pass-pipeline="builtin.module(func.func(tosa-to-linalg),empty-tensor-to-alloc-tensor,arith-bufferize, func.func(linalg-bufferize), func.func(tensor-bufferize), func-bufferize,func.func(buffer-deallocation, convert-linalg-to-loops) , convert-scf-to-cf, expand-strided-metadata, memref-expand)" | \
	${MLIR_OPT} -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin))' | \
	${MLIR_OPT} -gpu-async-region -gpu-to-llvm | \
	${MLIR_OPT} -async-to-async-runtime -async-runtime-ref-counting | \
	${MLIR_OPT} -convert-async-to-llvm -convert-func-to-llvm | \
	${MLIR_CPU_RUNNER} -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME} \
	-shared-libs=${MLIR_ASYNC_RUNTIME} ${OPT_FLAG}
