#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
LLC := ../../llvm/build/bin/llc
OPT_FLAG := -O0

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
MTRIPLE := x86_64-apple-darwin
endif

vector-exp-config-lower:
	@${BUDDY_OPT} ./vector-exp-predication.mlir \
		-lower-vector-exp \
		-convert-vector-to-llvm \
		-convert-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts \
		-o ./log.mlir

vector-exp-config-translate:
	@${BUDDY_OPT} ./vector-exp-predication.mlir \
		-lower-vector-exp \
		-convert-vector-to-llvm \
		-convert-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll

vector-exp-config-run:
	@${BUDDY_OPT} ./vector-exp-predication.mlir \
		-lower-vector-exp \
		-convert-vector-to-llvm \
		-convert-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
