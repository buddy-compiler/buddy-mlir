#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
MLIR_TOYC := ../../llvm/build/bin/toyc-ch7
BUDDY_TOY_DSL := ../../build-dsl/bin/buddy-toy-dsl

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

toyc-constant-run:
	@${MLIR_TOYC} ./constant.toy -emit=jit

buddy-toy-constant-ast:
	@${BUDDY_TOY_DSL} ./constant.toy  -emit=ast

buddy-toy-constant-mlir:
	@${BUDDY_TOY_DSL} ./constant.toy  -emit=mlir

buddy-toy-constant-affine:
	@${BUDDY_TOY_DSL} ./constant.toy  -emit=mlir-affine

buddy-toy-constant-llvm:
	@${BUDDY_TOY_DSL} ./constant.toy  -emit=mlir-llvm

buddy-toy-constant-translate:
	@${BUDDY_TOY_DSL} ./constant.toy  -emit=llvm

buddy-toy-constant-run:
	@${BUDDY_TOY_DSL} ./constant.toy  -emit=jit
