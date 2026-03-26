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

SOFTMAX_INPUT ?= ./softmax/linalg_bufferized_no_vec_no_loops.mlir
SOFTMAX_OUT_DIR ?= .
SOFTMAX_LINALG_TO_VIR_OUT ?= ${SOFTMAX_OUT_DIR}/log.softmax.after_lower_linalg_to_vir.mlir
SOFTMAX_VIR_TO_VECTOR_OUT ?= ${SOFTMAX_OUT_DIR}/log.softmax.after_lower_vir_to_vector.mlir
SOFTMAX_FINAL_LL_OUT ?= ${SOFTMAX_OUT_DIR}/log.softmax.final.ll

LAYERNORM_INPUT ?= ./layernorm/linalg_bufferized_no_vec_no_loops.mlir
LAYERNORM_OUT_DIR ?= .
LAYERNORM_LINALG_TO_VIR_OUT ?= ${LAYERNORM_OUT_DIR}/log.layernorm.after_lower_linalg_to_vir.mlir
LAYERNORM_VIR_TO_VECTOR_OUT ?= ${LAYERNORM_OUT_DIR}/log.layernorm.after_lower_vir_to_vector.mlir
LAYERNORM_FINAL_LL_OUT ?= ${LAYERNORM_OUT_DIR}/log.layernorm.final.ll

MATMUL_PIPELINE := -lower-linalg-to-vir -lower-vir-to-vector="vector-width=4" -cse \
	-convert-vector-to-scf -expand-strided-metadata -lower-affine \
	-convert-scf-to-cf -convert-cf-to-llvm -convert-vector-to-llvm -convert-math-to-llvm \
	-finalize-memref-to-llvm -convert-arith-to-llvm -convert-func-to-llvm \
	-reconcile-unrealized-casts

SOFTMAX_PIPELINE := ${MATMUL_PIPELINE}
LAYERNORM_PIPELINE := ${MATMUL_PIPELINE}

.PHONY: \
	all-case-study \
	matmul-case-study matmul-linalg-to-vir matmul-vir-to-vector matmul-llvm-ir matmul-run \
	softmax-case-study softmax-linalg-to-vir softmax-vir-to-vector softmax-llvm-ir softmax-run \
	layernorm-case-study layernorm-linalg-to-vir layernorm-vir-to-vector layernorm-llvm-ir layernorm-run

all-case-study: matmul-case-study softmax-case-study layernorm-case-study

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

softmax-case-study: softmax-linalg-to-vir softmax-vir-to-vector softmax-llvm-ir softmax-run

softmax-linalg-to-vir:
	@${BUDDY_OPT} ${SOFTMAX_INPUT} \
		-lower-linalg-to-vir \
		-o ${SOFTMAX_LINALG_TO_VIR_OUT}

softmax-vir-to-vector:
	@${BUDDY_OPT} ${SOFTMAX_INPUT} \
		-lower-linalg-to-vir \
		-lower-vir-to-vector="vector-width=4" \
		-o ${SOFTMAX_VIR_TO_VECTOR_OUT}

softmax-llvm-ir:
	@${BUDDY_OPT} ${SOFTMAX_INPUT} \
		${SOFTMAX_PIPELINE} | \
	${MLIR_TRANSLATE} --mlir-to-llvmir \
		-o ${SOFTMAX_FINAL_LL_OUT}

softmax-run:
	@${BUDDY_OPT} ${SOFTMAX_INPUT} \
		${SOFTMAX_PIPELINE} | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

layernorm-case-study: layernorm-linalg-to-vir layernorm-vir-to-vector layernorm-llvm-ir layernorm-run

layernorm-linalg-to-vir:
	@${BUDDY_OPT} ${LAYERNORM_INPUT} \
		-lower-linalg-to-vir \
		-o ${LAYERNORM_LINALG_TO_VIR_OUT}

layernorm-vir-to-vector:
	@${BUDDY_OPT} ${LAYERNORM_INPUT} \
		-lower-linalg-to-vir \
		-lower-vir-to-vector="vector-width=4" \
		-o ${LAYERNORM_VIR_TO_VECTOR_OUT}

layernorm-llvm-ir:
	@${BUDDY_OPT} ${LAYERNORM_INPUT} \
		${LAYERNORM_PIPELINE} | \
	${MLIR_TRANSLATE} --mlir-to-llvmir \
		-o ${LAYERNORM_FINAL_LL_OUT}

layernorm-run:
	@${BUDDY_OPT} ${LAYERNORM_INPUT} \
		${LAYERNORM_PIPELINE} | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
