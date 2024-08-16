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

gpu-test-lower:
	@${MLIR_OPT} forward.mlir -gpu-kernel-outlining -llvm-request-c-wrappers | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target{chip=sm_70 O=3},\
	gpu.module(convert-scf-to-cf, convert-gpu-to-nvvm, convert-arith-to-llvm), convert-scf-to-cf, gpu-to-llvm, gpu-module-to-binary)" | \
	${MLIR_OPT} -o log.mlir

gpu-test-translate:
	@${MLIR_OPT} forward.mlir -gpu-kernel-outlining -llvm-request-c-wrappers | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target{chip=sm_70 O=3},\
	gpu.module(convert-scf-to-cf, convert-gpu-to-nvvm, convert-arith-to-llvm), convert-scf-to-cf, gpu-to-llvm, gpu-module-to-binary)" | \
	${MLIR_TRANSLATE} -mlir-to-llvmir -o log.ll

gpu-test-run:
	@${MLIR_OPT} forward.mlir -gpu-kernel-outlining -llvm-request-c-wrappers | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target{chip=sm_70 O=3},\
	gpu.module(convert-scf-to-cf, convert-gpu-to-nvvm, convert-arith-to-llvm), convert-scf-to-cf, gpu-to-llvm, gpu-module-to-binary)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME} 
