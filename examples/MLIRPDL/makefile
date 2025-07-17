#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-runner
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

pdl-multiroot-lower:
	@${MLIR_OPT} ./pdl-multiroot.mlir \
		-allow-unregistered-dialect -test-pdl-bytecode-pass -split-input-file \
		-o ./log.mlir

pdl-bud-fma-lower:
	@${MLIR_OPT} ./pdl-bud-fma.mlir \
		-allow-unregistered-dialect -test-pdl-bytecode-pass --convert-vector-to-llvm \
		--finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts \
		-o ./log.mlir
