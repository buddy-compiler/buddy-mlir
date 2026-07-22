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
	        -mattr=+xboscame \
	        -o bosc-mma.s
	@echo "Assembly saved to bosc-mma.s"

linalg-matmul-lower:
	@${BUDDY_OPT} ./linalg-to-boscame-matmul.mlir \
		-lower-linalg-to-boscame \
		-o linalg-matmul-lowered.mlir

linalg-matmul-translate:
	@${BUDDY_OPT} ./linalg-to-boscame-matmul.mlir \
		-lower-linalg-to-boscame \
		-lower-bosc-ame \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-expand-strided-metadata \
		-lower-affine \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o linalg-matmul.ll

linalg-matmul-asm:
	@${BUDDY_OPT} ./linalg-to-boscame-matmul.mlir \
		-lower-linalg-to-boscame \
		-lower-bosc-ame \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-expand-strided-metadata \
		-lower-affine \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+xboscame \
		-o linalg-matmul.s

generic-matmul-lower:
	@${BUDDY_OPT} ./generic-matmul.mlir \
		-lower-linalg-to-boscame \
		-o generic-matmul-lowered.mlir

generic-matmul-translate:
	@${BUDDY_OPT} ./generic-matmul.mlir \
		-lower-linalg-to-boscame \
		-lower-bosc-ame \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-expand-strided-metadata \
		-lower-affine \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o generic-matmul.ll

generic-matmul-asm:
	@${BUDDY_OPT} ./generic-matmul.mlir \
		-lower-linalg-to-boscame \
		-lower-bosc-ame \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-expand-strided-metadata \
		-lower-affine \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+xboscame \
		-o generic-matmul.s

generic-elementwise-add-lower:
	@${BUDDY_OPT} ./generic-elementwise-add.mlir \
		-lower-linalg-to-boscame \
		-o generic-elementwise-add-lowered.mlir

generic-elementwise-add-translate:
	@${BUDDY_OPT} ./generic-elementwise-add.mlir \
		-lower-linalg-to-boscame \
		-lower-bosc-ame \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-expand-strided-metadata \
		-lower-affine \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o generic-elementwise-add.ll

generic-elementwise-add-asm:
	@${BUDDY_OPT} ./generic-elementwise-add.mlir \
		-lower-linalg-to-boscame \
		-lower-bosc-ame \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-expand-strided-metadata \
		-lower-affine \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+xboscame \
		-o generic-elementwise-add.s

generic-transpose-lower:
	@${BUDDY_OPT} ./generic-transpose.mlir \
		-lower-linalg-to-boscame \
		-o generic-transpose-lowered.mlir

generic-transpose-translate:
	@${BUDDY_OPT} ./generic-transpose.mlir \
		-lower-linalg-to-boscame \
		-lower-bosc-ame \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-expand-strided-metadata \
		-lower-affine \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o generic-transpose.ll

generic-transpose-asm:
	@${BUDDY_OPT} ./generic-transpose.mlir \
		-lower-linalg-to-boscame \
		-lower-bosc-ame \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-expand-strided-metadata \
		-lower-affine \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+xboscame \
		-o generic-transpose.s

generic-broadcast-row-lower:
	@${BUDDY_OPT} ./generic-broadcast-row.mlir \
		-lower-linalg-to-boscame \
		-o generic-broadcast-row-lowered.mlir

generic-broadcast-row-translate:
	@${BUDDY_OPT} ./generic-broadcast-row.mlir \
		-lower-linalg-to-boscame \
		-lower-bosc-ame \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-expand-strided-metadata \
		-lower-affine \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o generic-broadcast-row.ll

generic-broadcast-row-asm:
	@${BUDDY_OPT} ./generic-broadcast-row.mlir \
		-lower-linalg-to-boscame \
		-lower-bosc-ame \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-expand-strided-metadata \
		-lower-affine \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+xboscame \
		-o generic-broadcast-row.s

generic-broadcast-column-lower:
	@${BUDDY_OPT} ./generic-broadcast-column.mlir \
		-lower-linalg-to-boscame \
		-o generic-broadcast-column-lowered.mlir

generic-broadcast-column-translate:
	@${BUDDY_OPT} ./generic-broadcast-column.mlir \
		-lower-linalg-to-boscame \
		-lower-bosc-ame \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-expand-strided-metadata \
		-lower-affine \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o generic-broadcast-column.ll

generic-broadcast-column-asm:
	@${BUDDY_OPT} ./generic-broadcast-column.mlir \
		-lower-linalg-to-boscame \
		-lower-bosc-ame \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-expand-strided-metadata \
		-lower-affine \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+xboscame \
		-o generic-broadcast-column.s

generic-broadcast-element-lower:
	@${BUDDY_OPT} ./generic-broadcast-element.mlir \
		-lower-linalg-to-boscame \
		-o generic-broadcast-element-lowered.mlir

generic-broadcast-element-translate:
	@${BUDDY_OPT} ./generic-broadcast-element.mlir \
		-lower-linalg-to-boscame \
		-lower-bosc-ame \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-expand-strided-metadata \
		-lower-affine \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o generic-broadcast-element.ll

generic-broadcast-element-asm:
	@${BUDDY_OPT} ./generic-broadcast-element.mlir \
		-lower-linalg-to-boscame \
		-lower-bosc-ame \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-expand-strided-metadata \
		-lower-affine \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+xboscame \
		-o generic-broadcast-element.s

clean:
	rm -f *.mlir.out *.ll *.s *-lowered.mlir
