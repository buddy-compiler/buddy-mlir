#!/bin/bash
BUDDY_OPT := ../build/bin/buddy-opt
MLIR_OPT := ../llvm/build/bin/mlir-opt
MLIR_CPU_RUNNER := ../llvm/build/bin/mlir-cpu-runner
OPT_FLAG := -O3

STRIP := 64
FILTER_min := 3
FILTER_max := 19
FILTER_step := 4
OUTPUT_min := 64
OUTPUT_max := 1024
OUTPUT_step := 64

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
# Generate the test cases from the template.
$(shell ./gen.sh ${FILTER_min} ${FILTER_max} ${FILTER_step} ${OUTPUT_min} ${OUTPUT_max} ${OUTPUT_step})
SOURCE=$(shell ls tempFile/*.mlir -v)
OUT=$(patsubst tempFile/%.mlir, %.out, $(SOURCE))

.PHONY:all
all:$(OUT) 
	$(shell rm -rf tempFile)

BUDDY_OPT_OPTIONS := -conv-vectorization="strip-mining=${STRIP}" -lower-affine -convert-scf-to-cf -convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts
MLIR_OPT_OPTIONS := -convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts

$(OUT):$(SOURCE)
	@echo $*
	@${BUDDY_OPT} tempFile/$*.mlir ${BUDDY_OPT_OPTIONS} | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

	@${MLIR_OPT} tempFile/$*.mlir ${MLIR_OPT_OPTIONS} | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
	@echo
