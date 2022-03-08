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
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
MTRIPLE := x86_64-apple-darwin
endif

tensor-print-lower:
	@${MLIR_OPT} ./tensor-print.mlir \
		-arith-bufferize  -tensor-bufferize -linalg-bufferize \
		-func-bufferize -buffer-deallocation -convert-linalg-to-loops \
		-convert-linalg-to-llvm -convert-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts -o ./log.mlir

tensor-print-translate:
	@${MLIR_OPT} ./tensor-print.mlir \
		-arith-bufferize  -tensor-bufferize -linalg-bufferize \
		-func-bufferize -buffer-deallocation -convert-linalg-to-loops \
		-convert-linalg-to-llvm -convert-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

tensor-print-run:
	@${MLIR_OPT} ./tensor-print.mlir \
		-arith-bufferize  -tensor-bufferize -linalg-bufferize \
		-func-bufferize -buffer-deallocation -convert-linalg-to-loops \
		-convert-linalg-to-llvm -convert-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
