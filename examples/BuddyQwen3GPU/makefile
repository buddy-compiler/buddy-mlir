BUDDY_BUILD_DIR := ../../build
LLVM_BUILD_DIR  := ../../llvm/build

BUDDY_OPT      := ${BUDDY_BUILD_DIR}/bin/buddy-opt
MLIR_OPT       := ${LLVM_BUILD_DIR}/bin/mlir-opt
MLIR_TRANSLATE := ${LLVM_BUILD_DIR}/bin/mlir-translate
LLVM_AS        := ${LLVM_BUILD_DIR}/bin/llvm-as
LLC            := ${LLVM_BUILD_DIR}/bin/llc
CLANGXX        := ${LLVM_BUILD_DIR}/bin/clang++

MLIR_LIB := ${LLVM_BUILD_DIR}/lib

CUDA_SM       ?= sm_80
CUDA_FEATURES ?= +ptx71
NVVM_PIPELINE := cubin-chip=${CUDA_SM} cubin-features=${CUDA_FEATURES} cubin-format=fatbin
TOSA_PIPELINE := builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))

BUILD_PATH := $(shell pwd)/
CXXFLAGS := -std=c++17 -O2 -DQWEN3_NEXTGPU_BUILD_PATH=\"$(BUILD_PATH)\"
LDFLAGS  := -L${MLIR_LIB} -lmlir_cuda_runtime -lmlir_c_runner_utils -lmlir_runner_utils -ldl -lm -lpthread -Wl,-rpath,${MLIR_LIB}

.PHONY: all clean next-qwen3-kernels-obj next-qwen3-e2e-run

all: next-qwen3-e2e-run

next-qwen3-prefill-kernel-obj: next-qwen3-prefill-kernel.o

next-qwen3-prefill-kernel.o: next-qwen3-prefill-kernel.mlir
	@${BUDDY_OPT} $< -simplify-tosa-reshape | \
	${BUDDY_OPT} -pass-pipeline "${TOSA_PIPELINE}" | \
	${BUDDY_OPT} \
		"-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
		-buffer-deallocation-pipeline \
		-convert-linalg-to-parallel-loops \
		-canonicalize \
		-gpu-map-parallel-loops \
		-convert-parallel-loops-to-gpu \
		-gpu-kernel-outlining \
		-buffer-hoisting \
		-canonicalize -cse | \
	${BUDDY_OPT} \
		-convert-memcpy-to-gpu \
		-gpu-async-region \
		-canonicalize \
		-llvm-request-c-wrappers | \
	${MLIR_OPT} \
		"-gpu-lower-to-nvvm-pipeline=${NVVM_PIPELINE}" | \
	${MLIR_OPT} \
		-gpu-to-llvm \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir > next-qwen3-prefill-kernel.ll
	@${LLVM_AS} next-qwen3-prefill-kernel.ll -o next-qwen3-prefill-kernel.bc
	@${LLC} next-qwen3-prefill-kernel.bc -filetype=obj -relocation-model=pic -O3 -o $@

next-qwen3-decode-kernel-obj: next-qwen3-decode-kernel.o

next-qwen3-decode-kernel.o: next-qwen3-decode-kernel.mlir
	@${BUDDY_OPT} $< -simplify-tosa-reshape | \
	${BUDDY_OPT} -pass-pipeline "${TOSA_PIPELINE}" | \
	${BUDDY_OPT} \
		"-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
		-buffer-deallocation-pipeline \
		-convert-linalg-to-parallel-loops \
		-canonicalize \
		-gpu-map-parallel-loops \
		-convert-parallel-loops-to-gpu \
		-gpu-kernel-outlining \
		-buffer-hoisting \
		-canonicalize -cse | \
	${BUDDY_OPT} \
		-convert-memcpy-to-gpu \
		-gpu-async-region \
		-canonicalize \
		-llvm-request-c-wrappers | \
	${MLIR_OPT} \
		"-gpu-lower-to-nvvm-pipeline=${NVVM_PIPELINE}" | \
	${MLIR_OPT} \
		-gpu-to-llvm \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir > next-qwen3-decode-kernel.ll
	@${LLVM_AS} next-qwen3-decode-kernel.ll -o next-qwen3-decode-kernel.bc
	@${LLC} next-qwen3-decode-kernel.bc -filetype=obj -relocation-model=pic -O3 -o $@

next-qwen3-kernels-obj: next-qwen3-prefill-kernel.o next-qwen3-decode-kernel.o

next-qwen3-e2e-run: next-qwen3-kernels-obj buddy-qwen3-nextgpu-main.cpp
	@${CLANGXX} ${CXXFLAGS} buddy-qwen3-nextgpu-main.cpp \
		next-qwen3-prefill-kernel.o next-qwen3-decode-kernel.o \
		${LDFLAGS} -o buddy-qwen3-nextgpu-run
	@cp ../BuddyQwen3/vocab.txt . 2>/dev/null || true

clean:
	@rm -f *.ll *.bc *.o buddy-qwen3-nextgpu-run
