#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
LLC := ../../llvm/build/bin/llc
LLVM_AS := ../../llvm/build/bin/llvm-as
OPT_FLAG := -O0

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so
MLIR_ASYNC_RUNTIME := ../../llvm/build/lib/libmlir_async_runtime.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
MLIR_ASYNC_RUNTIME := ./../llvm/build/lib/libmlir_async_runtime.dylib
MTRIPLE := arm64-apple-darwin
endif


test:
	@${MLIR_OPT} llama.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-linalg-bufferize \
		-matmul-paralell-vectorization-optimize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-llvm-request-c-wrappers \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLVM_AS} | \
	${LLC} -mtriple=${MTRIPLE} -filetype=obj -o test.o
	g++ test.o llama-main.cpp -I/Users/plct/PLCT/limeng/buddy-mlir/frontend/Interfaces -lmlir_runner_utils -lmlir_c_runner_utils -lomp -o test


lower:
	@${MLIR_OPT} llama.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-linalg-bufferize \
		-matmul-paralell-vectorization-optimize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
	-o ./lower.mlir


lowertest:
	@${BUDDY_OPT} lower.mlir \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-llvm-request-c-wrappers \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-llvm-request-c-wrappers \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLVM_AS} | \
	${LLC} -mtriple=${MTRIPLE} -filetype=obj -o test.o
	g++ test.o test.cpp -I/root/buddy-mlir/frontend/Interfaces -lmlir_runner_utils -lmlir_c_runner_utils -o test
	
