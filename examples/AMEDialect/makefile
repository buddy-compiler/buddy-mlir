#!/usr/bin/env bash

BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
BUDDY_LLC := ../../build/bin/buddy-llc

.PHONY: all clean help \
	mqma-b-mm mqma-b-mm-lower mqma-b-mm-translate mqma-b-mm-asm \
	mma-w-mm mma-w-mm-lower mma-w-mm-translate mma-w-mm-asm \
	mma-dw-mm mma-dw-mm-lower mma-dw-mm-translate mma-dw-mm-asm \
	mma-complete mma-complete-lower mma-complete-translate mma-complete-asm

all: mqma-b-mm mma-w-mm mma-dw-mm



#===----------------------------------------------------------------------===#
# mqma.b.mm (int8 quad-widen matrix multiply: int8 -> int32)
#===----------------------------------------------------------------------===#

# Alias for backward compatibility
mqma-b-mm: mqma-b-mm-lower

mqma-b-mm-lower:
	@echo "=== Lowering AME mqma.b.mm (int8 -> int32 matrix multiply) ==="
	${BUDDY_OPT} mqma-b-mm.mlir \
	        --lower-ame \
	        -o mqma-b-mm-lowered.mlir
	@echo "Lowered MLIR saved to mqma-b-mm-lowered.mlir"

mqma-b-mm-translate:
	@echo "=== Translating mqma.b.mm to LLVM IR ==="
	@${BUDDY_OPT} mqma-b-mm.mlir \
	        --lower-ame \
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
	        -o mqma-b-mm.ll
	@echo "LLVM IR saved to mqma-b-mm.ll"

mqma-b-mm-asm:
	@echo "=== Generating assembly for mqma.b.mm ==="
	@${BUDDY_OPT} mqma-b-mm.mlir \
	        --lower-ame \
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
	        -o mqma-b-mm.s
	@echo "Assembly saved to mqma-b-mm.s"

#===----------------------------------------------------------------------===#
# mma.w.mm (int32 matrix multiply-accumulate)
#===----------------------------------------------------------------------===#

# Alias for backward compatibility
mma-w-mm: mma-w-mm-lower

mma-w-mm-lower:
	@echo "=== Lowering AME mma.w.mm (int32 matrix multiply-accumulate) ==="
	${BUDDY_OPT} mma-w-mm.mlir \
	        --lower-ame \
	        -o mma-w-mm-lowered.mlir
	@echo "Lowered MLIR saved to mma-w-mm-lowered.mlir"

mma-w-mm-translate:
	@echo "=== Translating mma.w.mm to LLVM IR ==="
	@${BUDDY_OPT} mma-w-mm.mlir \
	        --lower-ame \
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
	        -o mma-w-mm.ll
	@echo "LLVM IR saved to mma-w-mm.ll"

mma-w-mm-asm:
	@echo "=== Generating assembly for mma.w.mm ==="
	@${BUDDY_OPT} mma-w-mm.mlir \
	        --lower-ame \
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
	        -o mma-w-mm.s
	@echo "Assembly saved to mma-w-mm.s"

#===----------------------------------------------------------------------===#
# mma.dw.mm (int64 double-widen matrix multiply-accumulate: int32 -> int64)
#===----------------------------------------------------------------------===#

# Alias for backward compatibility
mma-dw-mm: mma-dw-mm-lower

mma-dw-mm-lower:
	@echo "=== Lowering AME mma.dw.mm (int32 -> int64 matrix multiply-accumulate) ==="
	${BUDDY_OPT} mma-dw-mm.mlir \
	        --lower-ame \
	        -o mma-dw-mm-lowered.mlir
	@echo "Lowered MLIR saved to mma-dw-mm-lowered.mlir"

mma-dw-mm-translate:
	@echo "=== Translating mma.dw.mm to LLVM IR ==="
	@${BUDDY_OPT} mma-dw-mm.mlir \
	        --lower-ame \
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
	        -o mma-dw-mm.ll
	@echo "LLVM IR saved to mma-dw-mm.ll"

mma-dw-mm-asm:
	@echo "=== Generating assembly for mma.dw.mm ==="
	@${BUDDY_OPT} mma-dw-mm.mlir \
	        --lower-ame \
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
	        -o mma-dw-mm.s
	@echo "Assembly saved to mma-dw-mm.s"

#===----------------------------------------------------------------------===#
# mma-complete (complete demo: config + load + compute + store)
#===----------------------------------------------------------------------===#

mma-complete: mma-complete-lower

mma-complete-lower:
	@echo "=== Lowering AME Complete MMA Demo ==="
	${BUDDY_OPT} mma-complete-demo.mlir \
	        --lower-ame \
	        -o mma-complete-demo-lowered.mlir
	@echo "Lowered MLIR saved to mma-complete-demo-lowered.mlir"

mma-complete-translate:
	@echo "=== Translating complete MMA demo to LLVM IR ==="
	@${BUDDY_OPT} mma-complete-demo.mlir \
	        --lower-ame \
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
	        -o mma-complete-demo.ll
	@echo "LLVM IR saved to mma-complete-demo.ll"

mma-complete-asm:
	@echo "=== Generating assembly for complete MMA demo ==="
	@${BUDDY_OPT} mma-complete-demo.mlir \
	        --lower-ame \
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
	        -o mma-complete-demo.s
	@echo "Assembly saved to mma-complete-demo.s"

clean:
	rm -f *.mlir.out *.ll *.s *-lowered.mlir
