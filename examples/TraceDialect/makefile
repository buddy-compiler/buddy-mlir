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

trace-conv2d-lower:
	@${BUDDY_OPT} ./trace-conv2d.mlir \
		--trace --lower-trace | \
    ${MLIR_OPT} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts \
		-o ./log.mlir

trace-conv2d-translate:
	@${BUDDY_OPT} ./trace-conv2d.mlir \
		--trace --lower-trace | \
	${MLIR_OPT} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

trace-conv2d-run:
	@${BUDDY_OPT} ./trace-conv2d.mlir \
		--lower-trace | \
	${MLIR_OPT} ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

