#!/bin/bash
BUDDY_BUILD_DIR := ../../build/
LLVM_BUILD_DIR := ../../llvm/build-gpu/
BUDDY_OPT := ${BUDDY_BUILD_DIR}/bin/buddy-opt
MLIR_OPT := ${LLVM_BUILD_DIR}/bin/mlir-opt
MLIR_CPU_RUNNER := ${LLVM_BUILD_DIR}/bin/mlir-runner
MLIR_TRANSLATE := ${LLVM_BUILD_DIR}/bin/mlir-translate
LLC := ${LLVM_BUILD_DIR}/bin/llc
CLANG := ${LLVM_BUILD_DIR}/bin/clang
OPT_FLAG := -O3

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.so
MLIR_ASYNC_RUNTIME := ${LLVM_BUILD_DIR}/lib/libmlir_async_runtime.so
MLIR_CUDA_RUNTIME := ${LLVM_BUILD_DIR}/lib/libmlir_cuda_runtime.so
MLIR_FLOAT16_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_float16_utils.so
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.dylib
MLIR_ASYNC_RUNTIME := ${LLVM_BUILD_DIR}/lib/libmlir_async_runtime.dylib
MLIR_CUDA_RUNTIME := ${LLVM_BUILD_DIR}/lib/libmlir_cuda_runtime.dylib
MLIR_FLOAT16_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_float16_utils.dylib
endif

gpu-all-reduce-and-ptx:
	@${MLIR_OPT} gpu-all-reduce-and.mlir \
		-gpu-lower-to-nvvm-pipeline \
		-debug-only=serialize-to-isa \
		> /dev/null 2> log.ptx

gpu-all-reduce-and-sass:
	@${MLIR_OPT} gpu-all-reduce-and.mlir \
		-gpu-lower-to-nvvm-pipeline \
		-debug-only=dump-sass \
		> /dev/null 2> log.sass

gpu-all-reduce-and-jit:
	@${MLIR_OPT} gpu-all-reduce-and.mlir \
		-gpu-lower-to-nvvm-pipeline | \
	${MLIR_CPU_RUNNER} -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_CUDA_RUNTIME}

gpu-all-reduce-and-aot:
	@${MLIR_OPT} gpu-all-reduce-and.mlir \
		-gpu-lower-to-nvvm-pipeline | \
	${MLIR_OPT} \
		-gpu-async-region \
		-gpu-to-llvm | \
	${MLIR_OPT} \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} \
		--mlir-to-llvmir \
		-o gpu-all-reduce-and.ll && \
	${CLANG} gpu-all-reduce-and.ll \
		${MLIR_RUNNER_UTILS} \
		${MLIR_CUDA_RUNTIME} \
		-no-pie \
		-Wl,-rpath,${LLVM_BUILD_DIR}/lib \
		-o gpu-all-reduce-and.out && \
	./gpu-all-reduce-and.out

gpu-vector-contract-ptx:
	@${MLIR_OPT} gpu-vector-contract.mlir \
		-convert-vector-to-llvm \
		-gpu-lower-to-nvvm-pipeline \
		-debug-only=serialize-to-isa \
		> /dev/null 2> log.ptx

gpu-vector-contract-sass:
	@${MLIR_OPT} gpu-vector-contract.mlir \
		-convert-vector-to-llvm \
		-gpu-lower-to-nvvm-pipeline \
		-debug-only=dump-sass \
		> /dev/null 2> log.sass

gpu-vector-contract-jit:
	@${MLIR_OPT} gpu-vector-contract.mlir \
		-convert-vector-to-llvm \
		-gpu-lower-to-nvvm-pipeline | \
	${MLIR_CPU_RUNNER} -entry-point-result=void \
		-shared-libs=${MLIR_CUDA_RUNTIME} \
		-shared-libs=${MLIR_C_RUNNER_UTILS} \
		-shared-libs=${MLIR_RUNNER_UTILS}

gpu-vector-contract-aot:
	@${MLIR_OPT} gpu-vector-contract.mlir \
		-convert-vector-to-llvm \
		-gpu-lower-to-nvvm-pipeline | \
	${MLIR_TRANSLATE} \
		--mlir-to-llvmir \
		-o vector-contract.ll && \
	${CLANG} vector-contract.ll \
		${MLIR_RUNNER_UTILS} \
		${MLIR_CUDA_RUNTIME} \
		${MLIR_C_RUNNER_UTILS} \
		${MLIR_FLOAT16_UTILS} \
		-no-pie \
		-Wl,-rpath,${LLVM_BUILD_DIR}/lib \
		-o vector-contract.out && \
	./vector-contract.out

gpu-tensor-core-ptx:
	@${MLIR_OPT} gpu-tensor-core.mlir \
		-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_86 \
		-debug-only=serialize-to-isa \
		> /dev/null 2> tensor-core.ptx

gpu-tensor-core-sass:
	@${MLIR_OPT} gpu-tensor-core.mlir \
		-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_86 \
		-debug-only=dump-sass \
		> /dev/null 2> tensor-core.sass

gpu-tensor-core-jit:
	@${MLIR_OPT} gpu-tensor-core.mlir \
		-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_86 | \
	${MLIR_CPU_RUNNER} -entry-point-result=void \
		-shared-libs=${MLIR_CUDA_RUNTIME} \
		-shared-libs=${MLIR_C_RUNNER_UTILS} \
		-shared-libs=${MLIR_RUNNER_UTILS}

gpu-tensor-core-aot:
	@${MLIR_OPT} gpu-tensor-core.mlir \
		-gpu-lower-to-nvvm-pipeline=cubin-chip=sm_86 | \
	${MLIR_TRANSLATE} \
		--mlir-to-llvmir \
		-o tensor-core.ll && \
	${CLANG} tensor-core.ll \
		${MLIR_RUNNER_UTILS} \
		${MLIR_CUDA_RUNTIME} \
		${MLIR_C_RUNNER_UTILS} \
		${MLIR_FLOAT16_UTILS} \
		-no-pie \
		-Wl,-rpath,${LLVM_BUILD_DIR}/lib \
		-o tensor-core.out && \
	./tensor-core.out

gpu-launch-func-lower:
	@${MLIR_OPT} gpu-launch-func.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin))' | \
	${MLIR_OPT}	-gpu-async-region -gpu-to-llvm -o log.mlir

gpu-launch-func-run:
	@${MLIR_OPT} gpu-launch-func.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin))' | \
	${MLIR_OPT}	-gpu-async-region -gpu-to-llvm | \
	${MLIR_CPU_RUNNER} -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

async-execute-lower:
	@${MLIR_OPT} async-execute.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin))' | \
	${MLIR_OPT}	-gpu-async-region -gpu-to-llvm | \
	${MLIR_OPT} -async-to-async-runtime -async-runtime-ref-counting | \
	${MLIR_OPT} -convert-async-to-llvm -convert-func-to-llvm -o log.mlir

async-execute-run:
	@${MLIR_OPT} async-execute.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin))' | \
	${MLIR_OPT}	-gpu-async-region -gpu-to-llvm | \
	${MLIR_OPT} -async-to-async-runtime -async-runtime-ref-counting | \
	${MLIR_OPT} -convert-async-to-llvm -convert-func-to-llvm | \
	${MLIR_CPU_RUNNER} -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME} \
	-shared-libs=${MLIR_ASYNC_RUNTIME} ${OPT_FLAG}

gpu-mma-lower:
	@${MLIR_OPT} gpu-mma.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin{chip=sm_70}))' | \
	${MLIR_OPT}	-convert-scf-to-cf -gpu-to-llvm -o log.mlir

gpu-mma-run:
	@${MLIR_OPT} gpu-mma.mlir -gpu-kernel-outlining | \
	${MLIR_OPT} -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin{chip=sm_70}))' | \
	${MLIR_OPT}	-convert-scf-to-cf -gpu-to-llvm | \
	${MLIR_CPU_RUNNER} -entry-point-result=void -shared-libs=${MLIR_CUDA_RUNTIME} -shared-libs=${MLIR_RUNNER_UTILS}

vector-to-gpu-lower:
	@${MLIR_OPT} vector-to-gpu.mlir -convert-linalg-to-loops | \
	${MLIR_OPT} -pass-pipeline="builtin.module(gpu.module(convert-vector-to-gpu,canonicalize))" | \
	${MLIR_OPT} -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin{chip=sm_70}))' | \
	${MLIR_OPT}	-convert-scf-to-cf -gpu-to-llvm -o log.mlir

vector-to-gpu-run:
	@${MLIR_OPT} vector-to-gpu.mlir -convert-linalg-to-loops | \
	${MLIR_OPT} -pass-pipeline="builtin.module(gpu.module(convert-vector-to-gpu,canonicalize))" | \
	${MLIR_OPT} -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin{chip=sm_70}))' | \
	${MLIR_OPT}	-convert-scf-to-cf -gpu-to-llvm | \
	${MLIR_CPU_RUNNER} -entry-point-result=void -shared-libs=${MLIR_CUDA_RUNTIME} -shared-libs=${MLIR_RUNNER_UTILS}
