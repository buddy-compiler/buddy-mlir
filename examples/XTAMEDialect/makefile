BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
BUDDY_LLC := ../../build/bin/buddy-llc

.PHONY: all clean help \
    ame-th-mmacc-w-b ame-th-mmacc-w-b-lower ame-th-mmacc-w-b-translate ame-th-mmacc-w-b-obj \
	ame-th-mmaccu-w-b ame-th-mmaccu-w-b-lower ame-th-mmaccu-w-b-translate ame-th-mmaccu-w-b-obj \
	ame-th-mfmacc-d-s ame-th-mfmacc-d-s-lower ame-th-mfmacc-d-s-translate ame-th-mfmacc-d-s-obj \
	ame-th-mfmacc-s ame-th-mfmacc-s-lower ame-th-mfmacc-s-translate ame-th-mfmacc-s-obj

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

ame-th-mmacc-w-b-obj:
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
	${BUDDY_LLC}  -filetype=obj -mtriple=riscv64-unknown-linux-gnu \
			-target-abi=lp64d \
	        -mattr=+m,+a,+f,+d,+c,+v,+xtheadame \
	        -o ame-th-mmacc-w-b.o
	@echo "Assembly saved to ame-th-mmacc-w-b.o"

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

ame-th-mmaccu-w-b-obj:
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
	${BUDDY_LLC}  -filetype=obj -mtriple=riscv64-unknown-linux-gnu \
	        -mattr=+m,+a,+f,+d,+c,+v,+xtheadame \
	        -o ame-th-mmaccu-w-b.o
	@echo "Assembly saved to ame-th-mmaccu-w-b.o"

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

ame-th-mfmacc-d-s-obj:
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
	${BUDDY_LLC}  -filetype=obj -mtriple=riscv64-unknown-linux-gnu \
			-target-abi=lp64d \
	        -mattr=+m,+a,+f,+d,+c,+v,+xtheadame \
	        -o ame-th-mfmacc-d-s.o
	@echo "Object file saved to ame-th-mfmacc-d-s.o"

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

ame-th-mfmacc-s-obj:
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
	${BUDDY_LLC}  -filetype=obj -mtriple=riscv64 \
			-target-abi=lp64d \
	        -mattr=+m,+a,+f,+d,+c,+v,+xtheadame \
	        -o ame-th-mfmacc-s.o
	@echo "Assembly saved to ame-th-mfmacc-s.o"

clean:
	rm -f *.ll *.o *-lowered.mlir
