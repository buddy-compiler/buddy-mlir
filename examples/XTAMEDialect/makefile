BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
BUDDY_LLC := ../../build/bin/buddy-llc-ext

.PHONY: all clean help \
    ame-th-mmacc-w-b ame-th-mmacc-w-b-lower ame-th-mmacc-w-b-translate ame-th-mmacc-w-b-asm


all: ame-th-mmacc-w-b


ame-th-mmacc-w-b: ame-th-mmacc-w-b-lower

ame-th-mmacc-w-b-lower:
	@echo "=== Lowering XTAME Tile-based MMACC with B operand demo ==="
	${BUDDY_OPT} ame-th-mmacc-w-b.mlir \
	        --lower-xt-ame \
	        -o ame-th-mmacc-w-b-lowered.mlir
	@echo "Lowered MLIR saved to ame-th-mmacc-w-b-lowered.mlir"

ame-th-mmacc-w-b-translate:
	@echo "=== Translating tile-based MMACC with B operand demo to LLVM IR ==="
	@${BUDDY_OPT} ame-th-mmacc-w-b.mlir \
	        --lower-xt-ame \
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
	        -o ame-th-mmacc-w-b.ll
	@echo "LLVM IR saved to ame-th-mmacc-w-b.ll"

ame-th-mmacc-w-b-asm:
	@echo "=== Generating assembly for tile-based MMACC with B operand demo ==="
	@${BUDDY_OPT} ame-th-mmacc-w-b.mlir \
	        --lower-xt-ame \
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
	${BUDDY_LLC}  -filetype=asm -mtriple=riscv64-unknown-linux-gnu \
	        -mattr=+m,+a,+c,+d,+buddyext \
	        -o ame-th-mmacc-w-b.s
	@echo "Assembly saved to ame-th-mmacc-w-b.s"

clean:
	rm -f *.ll *.s *-lowered.mlir
