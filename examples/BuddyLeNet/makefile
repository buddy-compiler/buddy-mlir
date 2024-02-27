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
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
MLIR_ASYNC_RUNTIME := ./../llvm/build/lib/libmlir_async_runtime.dylib
MTRIPLE := x86_64-apple-darwin
endif

buddy-lenet-lower:
	@${MLIR_OPT} ./fake-lenet.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" | \
		${MLIR_OPT} \
		-eliminate-empty-tensors \
		-convert-tensor-to-linalg \
		-linalg-bufferize \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-llvm-request-c-wrappers \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts \
		-o ./log.mlir

buddy-lenet-translate:
	@${MLIR_OPT} ./fake-lenet.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" | \
		${MLIR_OPT} \
		-eliminate-empty-tensors \
		-convert-tensor-to-linalg \
		-linalg-bufferize \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-llvm-request-c-wrappers \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir -o log.ll


buddy-lenet-run:
	@${MLIR_OPT} ./fake-lenet.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" | \
		${MLIR_OPT} \
		-eliminate-empty-tensors \
		-convert-tensor-to-linalg \
		-linalg-bufferize \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-llvm-request-c-wrappers \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
