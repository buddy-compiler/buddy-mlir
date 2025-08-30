#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-runner
LLC := ../../llvm/build/bin/llc
OPT_FLAG := -O0
CLANG := ../../llvm/build//bin/clang
MLIR_LIB := ../../llvm/build/lib/
BUDDY_LIB := ../../build/midend/lib/

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

cf-iteration-exit-lower:
	@${MLIR_OPT} ./cf-iteration-exit.mlir \
		-convert-vector-to-llvm \
    	-convert-cf-to-llvm \
		-convert-func-to-llvm \
		-convert-arith-to-llvm \
		-reconcile-unrealized-casts \
		-o ./log.mlir

cf-iteration-exit-translate:
	@${MLIR_OPT} ./cf-iteration-exit.mlir \
		-convert-vector-to-llvm \
    	-convert-cf-to-llvm \
		-convert-func-to-llvm \
		-convert-arith-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

cf-iteration-exit-run:
	@${MLIR_OPT} ./cf-iteration-exit.mlir \
		-convert-vector-to-llvm \
    	-convert-cf-to-llvm \
		-convert-func-to-llvm \
		-convert-arith-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
