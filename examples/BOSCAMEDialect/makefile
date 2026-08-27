#!/usr/bin/env bash

BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
BUDDY_LLC := ../../build/bin/buddy-llc

.PHONY: all clean help \
	bosc-mma bosc-mma-lower bosc-mma-translate bosc-mma-obj

all: bosc-mma

#===----------------------------------------------------------------------===#
# bosc-mma (BOSCAME MMA demo: config + load + compute + store)
#===----------------------------------------------------------------------===#

bosc-mma-lower:
	@echo "=== Lowering BOSCAME MMA Demo ==="
	${BUDDY_OPT} bosc-mma.mlir \
	        -lower-bosc-ame \
	        -o bosc-mma-lowered.mlir
	@echo "Lowered MLIR saved to bosc-mma-lowered.mlir"

bosc-mma-translate:
	@echo "=== Translating BOSCAME MMA demo to LLVM IR ==="
	@${BUDDY_OPT} bosc-mma.mlir \
	        -lower-bosc-ame \
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

bosc-mma-obj:
	@echo "=== Generating assembly for BOSCAME MMA demo ==="
	@${BUDDY_OPT} bosc-mma.mlir \
	        -lower-bosc-ame \
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
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
			-target-abi=lp64d \
	        -mattr=+m,+a,+f,+d,+c,+v,+xboscame \
	        -o bosc-mma.o
	@echo "Assembly saved to bosc-mma.o"

bosc-mfma-lower:
	@echo "=== Lowering BOSCAME MFMA Demo ==="
	${BUDDY_OPT} bosc-mfma.mlir \
	        -lower-bosc-ame \
	        -o bosc-mfma-lowered.mlir
	@echo "Lowered MLIR saved to bosc-mfma-lowered.mlir"

bosc-mfma-translate:
	@echo "=== Translating BOSCAME MFMA demo to LLVM IR ==="
	@${BUDDY_OPT} bosc-mfma.mlir \
	        -lower-bosc-ame \
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
	        -o bosc-mfma.ll
	@echo "LLVM IR saved to bosc-mfma.ll"

bosc-mfma-obj:
	@echo "=== Generating assembly for BOSCAME MFMA demo ==="
	@${BUDDY_OPT} bosc-mfma.mlir \
	        -lower-bosc-ame \
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
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
	        -target-abi=lp64d \
	        -mattr=+m,+a,+f,+d,+c,+v,+xboscame \
	        -o bosc-mfma.o
	@echo "Assembly saved to bosc-mfma.o"

bosc-mfmul-lower:
	@echo "=== Lowering BOSCAME MFMUL Demo ==="
	${BUDDY_OPT} bosc-mfmul.mlir \
	        -lower-bosc-ame \
	        -o bosc-mfmul-lowered.mlir
	@echo "Lowered MLIR saved to bosc-mfmul-lowered.mlir"

bosc-mfmul-translate:
	@echo "=== Translating BOSCAME MFMUL demo to LLVM IR ==="
	@${BUDDY_OPT} bosc-mfmul.mlir \
	        -lower-bosc-ame \
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
	        -o bosc-mfmul.ll
	@echo "LLVM IR saved to bosc-mfmul.ll"

bosc-mfmul-obj:
	@echo "=== Generating assembly for BOSCAME MFMUL demo ==="
	@${BUDDY_OPT} bosc-mfmul.mlir \
	        -lower-bosc-ame \
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
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
	        -target-abi=lp64d \
	        -mattr=+m,+a,+f,+d,+c,+v,+xboscame \
	        -o bosc-mfmul.o
	@echo "Assembly saved to bosc-mfmul.o"

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

linalg-matmul-obj:
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
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-target-abi=lp64d \
		-mattr=+m,+a,+f,+d,+c,+v,+xboscame \
		-o linalg-matmul.o

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

generic-matmul-obj:
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
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-target-abi=lp64d \
		-mattr=+m,+a,+f,+d,+c,+v,+xboscame \
		-o generic-matmul.o

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

generic-elementwise-add-obj:
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
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-target-abi=lp64d \
		-mattr=+m,+a,+f,+d,+c,+v,+xboscame \
		-o generic-elementwise-add.o

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

generic-transpose-obj:
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
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-target-abi=lp64d \
		-mattr=+m,+a,+f,+d,+c,+v,+xboscame \
		-o generic-transpose.o

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

generic-broadcast-row-obj:
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
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-target-abi=lp64d \
		-mattr=+m,+a,+f,+d,+c,+v,+xboscame \
		-o generic-broadcast-row.o

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

generic-broadcast-column-obj:
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
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-target-abi=lp64d \
		-mattr=+m,+a,+f,+d,+c,+v,+xboscame \
		-o generic-broadcast-column.o

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

generic-broadcast-element-obj:
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
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-target-abi=lp64d \
	    -mattr=+m,+a,+f,+d,+c,+v,+xboscame \
		-o generic-broadcast-element.o

generic-unary-square-f32-qwen-lower:
	@${BUDDY_OPT} ./generic-unary-square-f32-qwen.mlir \
		-lower-linalg-to-boscame \
		-o generic-unary-square-f32-qwen-lowered.mlir

generic-unary-square-f32-qwen-translate:
	@${BUDDY_OPT} ./generic-unary-square-f32-qwen.mlir \
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
		-o generic-unary-square-f32-qwen.ll

generic-unary-square-f32-qwen-obj:
	@${BUDDY_OPT} ./generic-unary-square-f32-qwen.mlir \
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
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-target-abi=lp64d \
	    -mattr=+m,+a,+f,+d,+c,+v,+xboscame \
		-o generic-unary-square-f32-qwen.o

clean:
	rm -f *.mlir.out *.ll *.o *-lowered.mlir
