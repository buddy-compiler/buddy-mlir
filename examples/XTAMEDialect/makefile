BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
BUDDY_LLC := ../../build/bin/buddy-llc-ext

.PHONY: all clean help \
    ame-th-mmacc-w-b ame-th-mmacc-w-b-lower ame-th-mmacc-w-b-translate ame-th-mmacc-w-b-asm \
	ame-th-mmaccu-w-b ame-th-mmaccu-w-b-lower ame-th-mmaccu-w-b-translate ame-th-mmaccu-w-b-asm \
	ame-th-mfmacc-d-s ame-th-mfmacc-d-s-lower ame-th-mfmacc-d-s-translate ame-th-mfmacc-d-s-asm \
	ame-th-mfmacc-s ame-th-mfmacc-s-lower ame-th-mfmacc-s-translate ame-th-mfmacc-s-asm

all: ame-th-mmacc-w-b ame-th-mmaccu-w-b ame-th-mfmacc-d-s ame-th-mfmacc-s


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

ame-th-mmaccu-w-b: ame-th-mmaccu-w-b-lower

ame-th-mmaccu-w-b-lower:
	@echo "=== Lowering XTAME Tile-based MMACC-U with B operand demo ==="
	${BUDDY_OPT} ame-th-mmaccu-w-b.mlir \
	        --lower-xt-ame \
	        -o ame-th-mmaccu-w-b-lowered.mlir
	@echo "Lowered MLIR saved to ame-th-mmaccu-w-b-lowered.mlir"

ame-th-mmaccu-w-b-translate:
	@echo "=== Translating tile-based MMACC-U with B operand demo to LLVM IR ==="
	@${BUDDY_OPT} ame-th-mmaccu-w-b.mlir \
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
	        -o ame-th-mmaccu-w-b.ll
	@echo "LLVM IR saved to ame-th-mmaccu-w-b.ll"

ame-th-mmaccu-w-b-asm:
	@echo "=== Generating assembly for tile-based MMACC-U with B operand demo ==="
	@${BUDDY_OPT} ame-th-mmaccu-w-b.mlir \
	        --lower-xt-ame \
	        -convert-linalg-to-loops \
	        -lower-affine \
	        -convert-scf-to-cf \
	        -convert-cf-to-llvm \
			-expand-strided-metadata \
	        -convert-arith-to-llvm \
	        -convert-math-to-llvm \
	        -convert-func-to-llvm \
	        -finalize-memref-to-llvm \
	        -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC}  -filetype=asm -mtriple=riscv64-unknown-linux-gnu \
	        -mattr=+m,+a,+c,+d,+buddyext \
	        -o ame-th-mmaccu-w-b.s
	@echo "Assembly saved to ame-th-mmaccu-w-b.s"

ame-th-mfmacc-d-s: ame-th-mfmacc-d-s-lower

ame-th-mfmacc-d-s-lower:
	@echo "=== Lowering XTAME Tile-based MFMACC with D and S operand demo ==="
	${BUDDY_OPT} ame-th-mfmacc-d-s.mlir \
	        --lower-xt-ame \
	        -o ame-th-mfmacc-d-s-lowered.mlir
	@echo "Lowered MLIR saved to ame-th-mfmacc-d-s-lowered.mlir"

ame-th-mfmacc-d-s-translate:
	@echo "=== Translating tile-based MFMACC with D and S operand demo to LLVM IR ==="
	@${BUDDY_OPT} ame-th-mfmacc-d-s.mlir \
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
	        -o ame-th-mfmacc-d-s.ll
	@echo "LLVM IR saved to ame-th-mfmacc-d-s.ll"

ame-th-mfmacc-d-s-asm:
	@echo "=== Generating assembly for tile-based MFMACC with D and S operand demo ==="
	@${BUDDY_OPT} ame-th-mfmacc-d-s.mlir \
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
	        -o ame-th-mfmacc-d-s.s
	@echo "Assembly saved to ame-th-mfmacc-d-s.s"

ame-th-mfmacc-s: ame-th-mfmacc-s-lower

ame-th-mfmacc-s-lower:
	@echo "=== Lowering XTAME Tile-based MFMACC with S operand demo ==="
	${BUDDY_OPT} ame-th-mfmacc-s.mlir \
	        --lower-xt-ame \
	        -o ame-th-mfmacc-s-lowered.mlir
	@echo "Lowered MLIR saved to ame-th-mfmacc-s-lowered.mlir"

ame-th-mfmacc-s-translate:
	@echo "=== Translating tile-based MFMACC with S operand demo to LLVM IR ==="
	@${BUDDY_OPT} ame-th-mfmacc-s.mlir \
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
	        -o ame-th-mfmacc-s.ll
	@echo "LLVM IR saved to ame-th-mfmacc-s.ll"

ame-th-mfmacc-s-asm:
	@echo "=== Generating assembly for tile-based MFMACC with S operand demo ==="
	@${BUDDY_OPT} ame-th-mfmacc-s.mlir \
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
	        -o ame-th-mfmacc-s.s
	@echo "Assembly saved to ame-th-mfmacc-s.s"

clean:
	rm -f *.ll *.s *-lowered.mlir
