#!/bin/bash
CONV_OPT := ../build/bin/conv-opt
MLIR_OPT := ../llvm/build/bin/mlir-opt
MLIR_CPU_RUNNER := ../llvm/build/bin/mlir-cpu-runner
OPT_FLAG := -O0

STRIP := 32
FILTER_min := 3
FILTER_max := 11
FILTER_step := 2
OUTPUT_min := 32
OUTPUT_max := 4096
OUTPUT_step := 0

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../llvm/build/lib/libmlir_c_runner_utils.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../llvm/build/lib/libmlir_c_runner_utils.dylib
MTRIPLE := x86_64-apple-darwin
endif

$(shell rm -rf tempFile)
$(shell ./gen.sh ${FILTER_min} ${FILTER_max} ${FILTER_step} ${OUTPUT_min} ${OUTPUT_max} ${OUTPUT_step})
SOURCE=$(shell ls tempFile/*.mlir -v)
OUT=$(patsubst tempFile/%.mlir, %.out, $(SOURCE))

.PHONY:all
all:$(OUT) 
	$(shell rm -rf tempFile)

TOOL_OPT := -conv-vectorization="strip-mining=${STRIP}" -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts

$(OUT):$(SOURCE)
	@echo $*
	@${CONV_OPT} tempFile/$*.mlir ${TOOL_OPT} | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

	@${MLIR_OPT} tempFile/$*.mlir -convert-linalg-to-loops -lower-affine -convert-linalg-to-llvm -convert-scf-to-std -convert-std-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
	@echo