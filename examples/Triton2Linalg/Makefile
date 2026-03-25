#!/bin/bash
BUDDY_BUILD_DIR := ../../build
LLVM_BUILD_DIR := ../../llvm/build
BUDDY_OPT := ${BUDDY_BUILD_DIR}/bin/buddy-opt
MLIR_TRANSLATE := ${LLVM_BUILD_DIR}/bin/mlir-translate
MLIR_CPU_RUNNER := ${LLVM_BUILD_DIR}/bin/mlir-runner
OPT_FLAG := -O0

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.so
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.dylib
endif

MATMUL_INPUT ?= ./matmul/linalg_bufferized_no_vec_no_loops.mlir
MATMUL_OUT_DIR ?= .
MATMUL_LINALG_TO_VIR_OUT ?= ${MATMUL_OUT_DIR}/log.matmul.after_lower_linalg_to_vir.mlir
MATMUL_VIR_TO_VECTOR_OUT ?= ${MATMUL_OUT_DIR}/log.matmul.after_lower_vir_to_vector.mlir
MATMUL_FINAL_LL_OUT ?= ${MATMUL_OUT_DIR}/log.matmul.final.ll

MATMUL_PIPELINE := -lower-linalg-to-vir -lower-vir-to-vector="vector-width=4" -cse \
	-convert-vector-to-scf -expand-strided-metadata -lower-affine \
	-convert-scf-to-cf -convert-cf-to-llvm -convert-vector-to-llvm -convert-math-to-llvm \
	-finalize-memref-to-llvm -convert-arith-to-llvm -convert-func-to-llvm \
	-reconcile-unrealized-casts

.PHONY: matmul-case-study matmul-linalg-to-vir matmul-vir-to-vector matmul-llvm-ir matmul-run matmul-check

matmul-case-study: matmul-linalg-to-vir matmul-vir-to-vector matmul-llvm-ir matmul-run

matmul-linalg-to-vir:
	@${BUDDY_OPT} ${MATMUL_INPUT} \
		-lower-linalg-to-vir \
		-o ${MATMUL_LINALG_TO_VIR_OUT}

matmul-vir-to-vector:
	@${BUDDY_OPT} ${MATMUL_INPUT} \
		-lower-linalg-to-vir \
		-lower-vir-to-vector="vector-width=4" \
		-o ${MATMUL_VIR_TO_VECTOR_OUT}

matmul-llvm-ir:
	@${BUDDY_OPT} ${MATMUL_INPUT} \
		${MATMUL_PIPELINE} | \
	${MLIR_TRANSLATE} --mlir-to-llvmir \
		-o ${MATMUL_FINAL_LL_OUT}

matmul-run:
	@${BUDDY_OPT} ${MATMUL_INPUT} \
		${MATMUL_PIPELINE} | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
