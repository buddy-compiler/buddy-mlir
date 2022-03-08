#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
LLC := ../../llvm/build/bin/llc
OPT_FLAG := -O0

# The JIT for RISC-V is not ready.

rvv-arith-lower:
	@${BUDDY_OPT} ./rvv-arith.mlir \
		--lower-rvv --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-arith-translate:
	@${BUDDY_OPT} ./rvv-arith.mlir \
		--lower-rvv --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll

rvv-masked-arith-lower:
	@${BUDDY_OPT} ./rvv-masked-arith.mlir \
		--lower-rvv --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-masked-arith-translate:
	@${BUDDY_OPT} ./rvv-masked-arith.mlir \
		--lower-rvv --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll

rvv-memory-lower:
	@${BUDDY_OPT} ./rvv-memory.mlir \
		--lower-rvv --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-memory-translate:
	@${BUDDY_OPT} ./rvv-memory.mlir \
		--lower-rvv --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll
