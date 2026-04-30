#!/usr/bin/env bash

BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
BUDDY_LLC := ../../build/bin/buddy-llc

.PHONY: all clean help \
	bosc-mma bosc-mma-lower bosc-mma-translate bosc-mma-asm

all: bosc-mma

#===----------------------------------------------------------------------===#
# bosc-mma (BOSCAME MMA demo: config + load + compute + store)
#===----------------------------------------------------------------------===#

bosc-mma: bosc-mma-lower

bosc-mma-lower:
	@echo "=== Lowering BOSCAME MMA Demo ==="
	${BUDDY_OPT} bosc-mma.mlir \
	        --lower-bosc-ame \
	        -o bosc-mma-lowered.mlir
	@echo "Lowered MLIR saved to bosc-mma-lowered.mlir"

bosc-mma-translate:
	@echo "=== Translating BOSCAME MMA demo to LLVM IR ==="
	@${BUDDY_OPT} bosc-mma.mlir \
	        --lower-bosc-ame \
	        -convert-linalg-to-loops \
	        -lower-affine \
	        -convert-scf-to-cf \
	        -convert-cf-to-llvm \
	        -convert-arith-to-llvm \
	        -convert-math-to-llvm \
	        -convert-func-to-llvm \
	        -finalize-memref-to-llvm \
	        -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir \
	        -o bosc-mma.ll
	@echo "LLVM IR saved to bosc-mma.ll"

bosc-mma-asm:
	@echo "=== Generating assembly for BOSCAME MMA demo ==="
	@${BUDDY_OPT} bosc-mma.mlir \
	        --lower-bosc-ame \
	        -convert-linalg-to-loops \
	        -lower-affine \
	        -convert-scf-to-cf \
	        -convert-cf-to-llvm \
	        -convert-arith-to-llvm \
	        -convert-math-to-llvm \
	        -convert-func-to-llvm \
	        -finalize-memref-to-llvm \
	        -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
	        -mattr=+m,+v,+buddyext \
	        -o bosc-mma.s
	@echo "Assembly saved to bosc-mma.s"

clean:
	rm -f *.mlir.out *.ll *.s *-lowered.mlir
