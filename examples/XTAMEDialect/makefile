BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
BUDDY_LLC := ../../build/bin/buddy-llc

LLVM_MC := ../../llvm/build/bin/llvm-mc
LD_LLD ?= ld.lld
QEMU_RISCV64 ?= qemu-riscv64
QEMU_CPU ?= xt-c9501fdvmt

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
	        -mattr=+xtheadame \
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
			-lower-affine \
	        -convert-arith-to-llvm \
	        -convert-math-to-llvm \
	        -convert-func-to-llvm \
	        -finalize-memref-to-llvm \
	        -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC}  -filetype=asm -mtriple=riscv64-unknown-linux-gnu \
	        -mattr=+xtheadame \
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
	        -mattr=+xtheadame \
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
	        -mattr=+xtheadame \
	        -o ame-th-mfmacc-s.s
	@echo "Assembly saved to ame-th-mfmacc-s.s"

#===----------------------------------------------------------------------===#
# linalg.matmul end to end: lowering, assembly, linking and a QEMU self-check.
# The generated program returns the number of mismatching output elements, so a
# correct run exits with status 0. Set QEMU_RISCV64 to the Xuantie QEMU build;
# the model must implement the XTAME instructions (`-cpu xt-c9501fdvmt`).
#===----------------------------------------------------------------------===#

linalg-matmul: linalg-matmul-lower

.PHONY: linalg-matmul linalg-matmul-lower linalg-matmul-asm linalg-matmul-run
linalg-matmul-lower:
	@echo "=== Lowering the end-to-end matmul to XTAME ==="
	${BUDDY_OPT} linalg-matmul.mlir --lower-linalg-to-xtame \
	        -o linalg-matmul-lowered.mlir
	@echo "Lowered MLIR saved to linalg-matmul-lowered.mlir"

linalg-matmul-asm:
	@echo "=== Generating assembly for the end-to-end matmul ==="
	@${BUDDY_OPT} linalg-matmul.mlir --lower-linalg-to-xtame \
	        -lower-xt-ame \
	        -convert-linalg-to-loops \
	        -lower-affine \
	        -convert-scf-to-cf \
	        -convert-cf-to-llvm \
	        -expand-strided-metadata \
	        -lower-affine \
	        -convert-arith-to-llvm \
	        -convert-math-to-llvm \
	        -convert-func-to-llvm \
	        -finalize-memref-to-llvm \
	        -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64-unknown-linux-gnu \
	        -mattr=+m,+a,+c,+d,+xtheadame \
	        -o linalg-matmul.s
	@echo "Assembly saved to linalg-matmul.s"

# The freestanding start stub is generated here so that no hand-written
# assembly needs to be checked in. It calls main and exits with its return
# value, which is the number of mismatching output elements.
linalg-matmul-start.s:
	@echo "=== Generating the freestanding start stub ==="
	@printf '%s\n' \
	        '  .text' \
	        '  .globl _start' \
	        '_start:' \
	        '  call main' \
	        '  li a7, 93' \
	        '  ecall' > linalg-matmul-start.s

linalg-matmul-run: linalg-matmul-asm linalg-matmul-start.s
	@echo "=== Linking and running the matmul under QEMU ==="
	${LLVM_MC} -triple=riscv64-unknown-linux-gnu -mattr=+xtheadame \
	        -filetype=obj linalg-matmul.s -o linalg-matmul.o
	${LLVM_MC} -triple=riscv64-unknown-linux-gnu \
	        -filetype=obj linalg-matmul-start.s \
	        -o linalg-matmul-start.o
	${LD_LLD} -m elf64lriscv -static -o linalg-matmul \
	        linalg-matmul-start.o linalg-matmul.o
	${QEMU_RISCV64} -cpu ${QEMU_CPU} ./linalg-matmul
	@echo "matmul self-check passed: all output elements matched"

clean:
	rm -f *.ll *-lowered.mlir linalg-matmul linalg-matmul.o \
	      ame-th-mmacc-w-b.s ame-th-mmaccu-w-b.s \
	      ame-th-mfmacc-d-s.s ame-th-mfmacc-s.s linalg-matmul.s \
	      linalg-matmul-start.s linalg-matmul-start.o
