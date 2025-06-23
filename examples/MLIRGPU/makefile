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
	@${MLIR_OPT} gpu-all-reduce-and.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target,\
	gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary)" -o log.mlir

gpu-all-reduce-and-run:
	@${MLIR_OPT} gpu-all-reduce-and.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target,\
	gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

gpu-launch-func-lower:
	@${MLIR_OPT} gpu-launch-func.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target,\
	gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary)" | \
	${MLIR_OPT}	-gpu-async-region -o log.mlir

gpu-launch-func-run:
	@${MLIR_OPT} gpu-launch-func.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target,\
	gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary)" | \
	${MLIR_OPT}	-gpu-async-region | \
	${MLIR_CPU_RUNNER} -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

async-execute-lower:
	@${MLIR_OPT} async-execute.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target,\
	gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary)" | \
	${MLIR_OPT}	-gpu-async-region | \
	${MLIR_OPT} -async-to-async-runtime -async-runtime-ref-counting | \
	${MLIR_OPT} -convert-async-to-llvm -convert-func-to-llvm -o log.mlir

async-execute-run:
	@${MLIR_OPT} async-execute.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(gpu.module(convert-gpu-to-nvvm), nvvm-attach-target)" | \
	${MLIR_OPT} -gpu-async-region -gpu-to-llvm -gpu-module-to-binary |\
	${MLIR_OPT} -async-to-async-runtime -async-runtime-ref-counting -convert-async-to-llvm -convert-func-to-llvm |\
	${MLIR_CPU_RUNNER} -shared-libs=${MLIR_CUDA_RUNTIME} -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_ASYNC_RUNTIME} -entry-point-result=void -O0

gpu-mma-lower:
	@${MLIR_OPT} gpu-mma.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target{chip=sm_70, O=3}, gpu.module(convert-gpu-to-nvvm))" | \
	${MLIR_OPT}	-convert-scf-to-cf -gpu-to-llvm -o log.mlir
		
gpu-mma-run:
	@${MLIR_OPT} gpu-mma.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(gpu.module(strip-debuginfo, convert-gpu-to-nvvm, convert-nvvm-to-llvm), nvvm-attach-target)" | \
	${MLIR_OPT}	-gpu-module-to-binary -convert-scf-to-cf -convert-cf-to-llvm -gpu-to-llvm | \
	${MLIR_CPU_RUNNER} -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME} \
	-shared-libs=${MLIR_ASYNC_RUNTIME} -entry-point-result=void -O0

vector-to-gpu-lower:
	@${MLIR_OPT} vector-to-gpu.mlir -convert-linalg-to-loops | \
	${MLIR_OPT} -pass-pipeline="builtin.module(gpu.module(convert-vector-to-gpu,canonicalize))" | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target{chip=sm_70 O=3},\
	gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary)" | \
	${MLIR_OPT}	-convert-scf-to-cf -o log.mlir 

vector-to-gpu-run:
	@${MLIR_OPT} vector-to-gpu.mlir -convert-linalg-to-loops | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target,\
	gpu.module(convert-gpu-to-nvvm, convert-vector-to-gpu), gpu-module-to-binary)" | \
	${MLIR_OPT}	-convert-scf-to-cf -gpu-to-llvm -reconcile-unrealized-casts| \
	${MLIR_CPU_RUNNER} -shared-libs=${MLIR_CUDA_RUNTIME} -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_ASYNC_RUNTIME} -entry-point-result=void -O3

gpu-memcpy-lower:
	@${MLIR_OPT} gpu-memcpy.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(gpu.module(convert-gpu-to-nvvm), nvvm-attach-target)" | \
	${MLIR_OPT} -gpu-async-region -gpu-to-llvm -gpu-module-to-binary |\
	${MLIR_OPT} -async-to-async-runtime -async-runtime-ref-counting -convert-async-to-llvm -convert-func-to-llvm -o log.mlir

gpu-memcpy-translate:
	@${MLIR_OPT} gpu-memcpy.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(gpu.module(convert-gpu-to-nvvm), nvvm-attach-target)" | \
	${MLIR_OPT} -gpu-async-region -gpu-to-llvm -gpu-module-to-binary |\
	${MLIR_OPT} -async-to-async-runtime -async-runtime-ref-counting -convert-async-to-llvm -convert-func-to-llvm |\
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

gpu-memcpy-run:
	@${MLIR_OPT} gpu-memcpy.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(gpu.module(convert-gpu-to-nvvm), nvvm-attach-target)" | \
	${MLIR_OPT} -gpu-async-region -gpu-to-llvm -gpu-module-to-binary |\
	${MLIR_OPT} -async-to-async-runtime -async-runtime-ref-counting -convert-async-to-llvm -convert-func-to-llvm |\
	${MLIR_CPU_RUNNER} -shared-libs=${MLIR_CUDA_RUNTIME} -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_ASYNC_RUNTIME} -entry-point-result=void -O0

gpu-memset-lower:
	@${MLIR_OPT} gpu-memset.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target,\
	gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary)" -o log.mlir\

gpu-memset-translate:
	@${MLIR_OPT} gpu-memset.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target,\
	gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary)"  |\
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

gpu-memset-run:
	@${MLIR_OPT} gpu-memset.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target,\
	gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary)" |\
	${MLIR_CPU_RUNNER} -entry-point-result=void -shared-libs=${MLIR_CUDA_RUNTIME} -shared-libs=${MLIR_RUNNER_UTILS}

gpu-shuffle-lower:
	@${MLIR_OPT} gpu-shuffle.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target,\
	gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary)" -o log.mlir

gpu-shuffle-translate:
	@${MLIR_OPT} gpu-shuffle.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target,\
	gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, convert-nvvm-to-llvm, gpu-module-to-binary)" |\
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

gpu-shuffle-run:
	@${MLIR_OPT} gpu-shuffle.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target,\
	gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

gpu-printf-lower:
	@${MLIR_OPT} gpu-printf.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target,\
	gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary)" -o log.mlir

gpu-printf-translate:
	@${MLIR_OPT} gpu-printf.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target,\
	gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, convert-nvvm-to-llvm, gpu-module-to-binary)" |\
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

gpu-printf-run:
	@${MLIR_OPT} gpu-printf.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target,\
	gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

gpu-multiple-all-reduce-lower:
	@${MLIR_OPT} gpu-multiple-all-reduce.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target,\
	gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary)" -o log.mlir

gpu-multiple-all-reduce-translate:
	@${MLIR_OPT} gpu-multiple-all-reduce.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target,\
	gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

gpu-multiple-all-reduce-run:
	@${MLIR_OPT} gpu-multiple-all-reduce.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline="builtin.module(nvvm-attach-target,\
	gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}