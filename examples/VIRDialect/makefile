#!/bin/bash
BUDDY_BUILD_DIR := ../../build/
LLVM_BUILD_DIR := ../../llvm/build/
BUDDY_OPT := ${BUDDY_BUILD_DIR}/bin/buddy-opt
MLIR_OPT := ${LLVM_BUILD_DIR}/bin/mlir-opt

vir-setvl-lower:
	${BUDDY_OPT} vir-setvl.mlir -o log.mlir

vir-dynamic-vec-type-lower:
	${BUDDY_OPT} vir-dynamic-vec-type.mlir \
	--allow-unregistered-dialect \
	-o log.mlir

vir-memory-lower:
	${BUDDY_OPT} vir-memory.mlir \
	-o log.mlir

vir-basic-lower:
	${BUDDY_OPT} vir-basic.mlir \
	-o log.mlir

vir-arith-lower:
	${BUDDY_OPT} vir-arith.mlir \
	-o log.mlir
