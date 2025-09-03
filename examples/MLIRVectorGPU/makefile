#!/bin/bash
BUDDY_BUILD_DIR := ../../build/
LLVM_BUILD_DIR := ../../llvm/build/
BUDDY_OPT := ${BUDDY_BUILD_DIR}/bin/buddy-opt
MLIR_OPT := ${LLVM_BUILD_DIR}/bin/mlir-opt
MLIR_TRANSLATE := ${LLVM_BUILD_DIR}/bin/mlir-translate
MLIR_CPU_RUNNER := ${LLVM_BUILD_DIR}/bin/mlir-runner
LLC := ${LLVM_BUILD_DIR}/bin/llc
OPT_FLAG := -O0
CUDA_COMPUTE_CAPACITY ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk 'NR==1{printf "sm_%.0f", $$0*10}')

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so
MLIR_CUDA_RUNTIME := ../../llvm/build/lib/libmlir_cuda_runtime.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
MTRIPLE := x86_64-apple-darwin
endif

.SECONDEXPANSION:
all-run: $$(run-targets)

# vector-load-lower:
vector-load-store-lower:
	@${MLIR_OPT} ./vector-load-store.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		arith-bufferize, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" \
	-o log.mlir

vector-load-store-translate:
	@${MLIR_OPT} ./vector-load-store.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		arith-bufferize, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

run-targets += vector-load-store-run
vector-load-store-run:
	@${MLIR_OPT} ./vector-load-store.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		arith-bufferize, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void \
	-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

run-targets += vector-load-store-run

vector-bitcast-lower:
	@${MLIR_OPT} ./vector-bitcast.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=sm_86 O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" \
	-o log.mlir

vector-bitcast-translate:
	@${MLIR_OPT} ./vector-bitcast.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=sm_86 O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

run-targets += vector-bitcast-run
vector-bitcast-run:
	@${MLIR_OPT} ./vector-bitcast.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=sm_86 O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void \
	-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}


vector-compressstore-lower:
	@${MLIR_OPT} ./vector-compressstore.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" \
	-o log.mlir

vector-compressstore-translate:
	@${MLIR_OPT} ./vector-compressstore.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

run-targets += vector-compressstore-run
vector-compressstore-run:
	@${MLIR_OPT} ./vector-compressstore.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void \
	-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

# The order of convert-arith-to-llvm and convert-vector-to-llvm matters.
vector-constant-mask-lower:
	@${MLIR_OPT} ./vector-constant-mask.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-arith-to-llvm, \
		convert-vector-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" \
	-o log.mlir

vector-constant-mask-translate:
	@${MLIR_OPT} ./vector-constant-mask.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-arith-to-llvm, \
		convert-vector-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

run-targets += vector-constant-mask-run
vector-constant-mask-run:
	@${MLIR_OPT} ./vector-constant-mask.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-arith-to-llvm, \
		convert-vector-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void \
	-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

vector-contract-lower:
	@${MLIR_OPT} ./vector-contract.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" \
	-o log.mlir

vector-contract-translate:
	@${MLIR_OPT} ./vector-contract.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

run-targets += vector-contract-run
vector-contract-run:
	@${MLIR_OPT} ./vector-contract.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void \
	-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

vector-create-mask-lower:
	@${MLIR_OPT} ./vector-create-mask.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" \
	-o log.mlir

vector-create-mask-translate:
	@${MLIR_OPT} ./vector-create-mask.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

run-targets += vector-create-mask-run
vector-create-mask-run:
	@${MLIR_OPT} ./vector-create-mask.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void \
	-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

vector-extract-lower:
	@${MLIR_OPT} ./vector-extract.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" \
	-o log.mlir

vector-extract-translate:
	@${MLIR_OPT} ./vector-extract.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

run-targets += vector-extract-run
vector-extract-run:
	@${MLIR_OPT} ./vector-extract.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void \
	-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

vector-fma-lower:
	@${MLIR_OPT} ./vector-fma.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" \
	-o log.mlir

vector-fma-translate:
	@${MLIR_OPT} ./vector-fma.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

run-targets += vector-fma-run
vector-fma-run:
	@${MLIR_OPT} ./vector-fma.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void \
	-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

vector-splat-lower:
	@${MLIR_OPT} ./vector-splat.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" \
	-o log.mlir

vector-splat-translate:
	@${MLIR_OPT} ./vector-splat.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

run-targets += vector-splat-run
vector-splat-run:
	@${MLIR_OPT} ./vector-splat.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void \
	-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

vector-gather-lower:
	@${MLIR_OPT} ./vector-gather.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" \
	-o log.mlir

vector-gather-translate:
	@${MLIR_OPT} ./vector-gather.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

run-targets += vector-gather-run
vector-gather-run:
	@${MLIR_OPT} ./vector-gather.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void \
	-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

vector-insert-lower:
	@${MLIR_OPT} ./vector-insert.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" \
	-o log.mlir

vector-insert-translate:
	@${MLIR_OPT} ./vector-insert.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

run-targets += vector-insert-run
vector-insert-run:
	@${MLIR_OPT} ./vector-insert.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void \
	-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

vector-transpose-lower:
	@${MLIR_OPT} ./vector-transpose.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" \
	-o log.mlir

vector-transpose-translate:
	@${MLIR_OPT} ./vector-transpose.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

run-targets += vector-transpose-run
vector-transpose-run:
	@${MLIR_OPT} ./vector-transpose.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void \
	-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

vector-outerproduct-lower:
	@${MLIR_OPT} ./vector-outerproduct.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" \
	-o log.mlir

vector-outerproduct-translate:
	@${MLIR_OPT} ./vector-outerproduct.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

run-targets += vector-outerproduct-run
vector-outerproduct-run:
	@${MLIR_OPT} ./vector-outerproduct.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void \
	-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

vector-reduction-lower:
	@${MLIR_OPT} ./vector-reduction.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" \
	-o log.mlir

vector-reduction-translate:
	@${MLIR_OPT} ./vector-reduction.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

run-targets += vector-reduction-run
vector-reduction-run:
	@${MLIR_OPT} ./vector-reduction.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void \
	-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

vector-type-cast-lower:
	@${MLIR_OPT} ./vector-type-cast.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" \
	-o log.mlir

vector-type-cast-translate:
	@${MLIR_OPT} ./vector-type-cast.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

run-targets += vector-type-cast-run
vector-type-cast-run:
	@${MLIR_OPT} ./vector-type-cast.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void \
	-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}

vector-shape-cast-lower:
	@${MLIR_OPT} ./vector-shape-cast.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" \
	-o log.mlir

vector-shape-cast-translate:
	@${MLIR_OPT} ./vector-shape-cast.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

run-targets += vector-shape-cast-run
vector-shape-cast-run:
	@${MLIR_OPT} ./vector-shape-cast.mlir \
	--pass-pipeline="builtin.module( \
		convert-linalg-to-loops, \
		convert-vector-to-scf, \
		lower-affine, \
		convert-scf-to-cf, \
		convert-vector-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		gpu-kernel-outlining, \
		nvvm-attach-target{chip=${CUDA_COMPUTE_CAPACITY} O=3}, \
		strip-debuginfo, \
		gpu.module(convert-gpu-to-nvvm), \
		gpu-to-llvm, \
		reconcile-unrealized-casts, \
		gpu-module-to-binary \
	)" | \
	${MLIR_CPU_RUNNER} -entry-point-result=void \
	-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}
