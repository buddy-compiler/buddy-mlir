SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

BUDDY_OPT ?= ../../build/bin/buddy-opt
BUDDY_TRANSLATE ?= ../../build/bin/buddy-translate
BUDDY_LLC ?= ../../build/bin/buddy-llc

.PHONY: all clean help \
	ame-matmul-lower ame-matmul-translate ame-matmul-asm ame-matmul-obj \
	ame-matmul-4x4-lower ame-matmul-4x4-translate \
	ame-matmul-4x4-asm ame-matmul-4x4-obj \
	ame-matmul-demo-lower ame-matmul-demo-translate \
	ame-matmul-demo-asm ame-matmul-demo-obj \
	linalg-to-ame-matmul-lower linalg-to-ame-matmul-translate \
	linalg-to-ame-matmul-asm linalg-to-ame-matmul-obj \
	linalg-to-ame-generic-lower linalg-to-ame-generic-translate \
	linalg-to-ame-generic-asm linalg-to-ame-generic-obj

all: ame-matmul-asm

#===----------------------------------------------------------------------===#
# ame-matmul (ZTT v0.6: configure + load + compute + store)
# Copy this four-target block when adding another example.
#===----------------------------------------------------------------------===#

ame-matmul-lower:
	@echo "=== Lowering AME matrix multiply-accumulate demo ==="
	${BUDDY_OPT} ame-matmul.mlir \
		--lower-ame \
		-o ame-matmul-lowered.mlir
	@echo "Lowered MLIR saved to ame-matmul-lowered.mlir"

ame-matmul-translate:
	@echo "=== Translating AME matrix multiply-accumulate demo to LLVM IR ==="
	@${BUDDY_OPT} ame-matmul.mlir \
		--lower-ame \
		--convert-linalg-to-loops \
		--lower-affine \
		--convert-scf-to-cf \
		--expand-strided-metadata \
		--lower-affine \
		--convert-cf-to-llvm \
		--convert-arith-to-llvm \
		--convert-math-to-llvm \
		--convert-func-to-llvm \
		--finalize-memref-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir \
		-o ame-matmul.ll
	@echo "Output saved to ame-matmul.ll"

ame-matmul-asm:
	@echo "=== Generating XAME assembly ==="
	@${BUDDY_OPT} ame-matmul.mlir \
		--lower-ame \
		--convert-linalg-to-loops \
		--lower-affine \
		--convert-scf-to-cf \
		--expand-strided-metadata \
		--lower-affine \
		--convert-cf-to-llvm \
		--convert-arith-to-llvm \
		--convert-math-to-llvm \
		--convert-func-to-llvm \
		--finalize-memref-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+xame -verify-machineinstrs \
		-o ame-matmul.s
	@echo "Output saved to ame-matmul.s"

ame-matmul-obj:
	@echo "=== Generating a RISC-V XAME object ==="
	@${BUDDY_OPT} ame-matmul.mlir \
		--lower-ame \
		--convert-linalg-to-loops \
		--lower-affine \
		--convert-scf-to-cf \
		--expand-strided-metadata \
		--lower-affine \
		--convert-cf-to-llvm \
		--convert-arith-to-llvm \
		--convert-math-to-llvm \
		--convert-func-to-llvm \
		--finalize-memref-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+m,+xame -verify-machineinstrs \
		-o ame-matmul.o
	@echo "Output saved to ame-matmul.o"

#===----------------------------------------------------------------------===#
# ame-matmul-4x4 (ZTT v0.6: AME_gem5-compatible profile)
#===----------------------------------------------------------------------===#

ame-matmul-4x4-lower:
	@echo "=== Lowering AME 4x4 matrix multiply-accumulate demo ==="
	${BUDDY_OPT} ame-matmul-4x4.mlir \
		--lower-ame \
		-o ame-matmul-4x4-lowered.mlir
	@echo "Lowered MLIR saved to ame-matmul-4x4-lowered.mlir"

ame-matmul-4x4-translate:
	@echo "=== Translating AME 4x4 matrix multiply-accumulate demo to LLVM IR ==="
	@${BUDDY_OPT} ame-matmul-4x4.mlir \
		--lower-ame \
		--convert-linalg-to-loops \
		--lower-affine \
		--convert-scf-to-cf \
		--expand-strided-metadata \
		--lower-affine \
		--convert-cf-to-llvm \
		--convert-arith-to-llvm \
		--convert-math-to-llvm \
		--convert-func-to-llvm \
		--finalize-memref-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir \
		-o ame-matmul-4x4.ll
	@echo "Output saved to ame-matmul-4x4.ll"

ame-matmul-4x4-asm:
	@echo "=== Generating XAME AME_gem5-profile assembly ==="
	@${BUDDY_OPT} ame-matmul-4x4.mlir \
		--lower-ame \
		--convert-linalg-to-loops \
		--lower-affine \
		--convert-scf-to-cf \
		--expand-strided-metadata \
		--lower-affine \
		--convert-cf-to-llvm \
		--convert-arith-to-llvm \
		--convert-math-to-llvm \
		--convert-func-to-llvm \
		--finalize-memref-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+xame,+xame-gem5 -verify-machineinstrs \
		-o ame-matmul-4x4.s
	@echo "Output saved to ame-matmul-4x4.s"

ame-matmul-4x4-obj:
	@echo "=== Generating an AME_gem5-profile RISC-V object ==="
	@${BUDDY_OPT} ame-matmul-4x4.mlir \
		--lower-ame \
		--convert-linalg-to-loops \
		--lower-affine \
		--convert-scf-to-cf \
		--expand-strided-metadata \
		--lower-affine \
		--convert-cf-to-llvm \
		--convert-arith-to-llvm \
		--convert-math-to-llvm \
		--convert-func-to-llvm \
		--finalize-memref-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+m,+xame,+xame-gem5 -verify-machineinstrs \
		-o ame-matmul-4x4.o
	@echo "Output saved to ame-matmul-4x4.o"

#===----------------------------------------------------------------------===#
# ame-matmul-demo (initialize + AME compute + print + scalar verification)
#===----------------------------------------------------------------------===#

ame-matmul-demo-lower:
	${BUDDY_OPT} ame-matmul-demo.mlir --lower-ame \
		-o ame-matmul-demo-lowered.mlir

ame-matmul-demo-translate:
	@${BUDDY_OPT} ame-matmul-demo.mlir \
		--lower-ame \
		--convert-linalg-to-loops \
		--lower-affine \
		--convert-scf-to-cf \
		--expand-strided-metadata \
		--lower-affine \
		--convert-cf-to-llvm \
		--convert-arith-to-llvm \
		--convert-math-to-llvm \
		--convert-func-to-llvm \
		--finalize-memref-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir -o ame-matmul-demo.ll

ame-matmul-demo-asm:
	@${BUDDY_OPT} ame-matmul-demo.mlir \
		--lower-ame \
		--convert-linalg-to-loops \
		--lower-affine \
		--convert-scf-to-cf \
		--expand-strided-metadata \
		--lower-affine \
		--convert-cf-to-llvm \
		--convert-arith-to-llvm \
		--convert-math-to-llvm \
		--convert-func-to-llvm \
		--finalize-memref-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 -target-abi=lp64 \
		-mattr=+m,+xame,+xame-gem5 -verify-machineinstrs \
		-o ame-matmul-demo.s

ame-matmul-demo-obj:
	@${BUDDY_OPT} ame-matmul-demo.mlir \
		--lower-ame \
		--convert-linalg-to-loops \
		--lower-affine \
		--convert-scf-to-cf \
		--expand-strided-metadata \
		--lower-affine \
		--convert-cf-to-llvm \
		--convert-arith-to-llvm \
		--convert-math-to-llvm \
		--convert-func-to-llvm \
		--finalize-memref-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 -target-abi=lp64 \
		-mattr=+m,+xame,+xame-gem5 -verify-machineinstrs \
		-o ame-matmul-demo.o

#===----------------------------------------------------------------------===#
# linalg-to-ame-matmul (Linalg matmul + AME lowering + scalar verification)
#===----------------------------------------------------------------------===#

linalg-to-ame-matmul-lower:
	${BUDDY_OPT} linalg-to-ame-matmul.mlir \
		--lower-linalg-to-ame='target=gem5' \
		-o linalg-to-ame-matmul-lowered.mlir

linalg-to-ame-matmul-translate:
	@${BUDDY_OPT} linalg-to-ame-matmul.mlir \
		--lower-linalg-to-ame='target=gem5' \
		--lower-ame \
		--convert-linalg-to-loops \
		--lower-affine \
		--convert-scf-to-cf \
		--expand-strided-metadata \
		--lower-affine \
		--convert-cf-to-llvm \
		--convert-arith-to-llvm \
		--convert-math-to-llvm \
		--convert-func-to-llvm \
		--finalize-memref-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir -o linalg-to-ame-matmul.ll

linalg-to-ame-matmul-asm:
	@${BUDDY_OPT} linalg-to-ame-matmul.mlir \
		--lower-linalg-to-ame='target=gem5' \
		--lower-ame \
		--convert-linalg-to-loops \
		--lower-affine \
		--convert-scf-to-cf \
		--expand-strided-metadata \
		--lower-affine \
		--convert-cf-to-llvm \
		--convert-arith-to-llvm \
		--convert-math-to-llvm \
		--convert-func-to-llvm \
		--finalize-memref-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 -target-abi=lp64 \
		-mattr=+m,+xame,+xame-gem5 -verify-machineinstrs \
		-o linalg-to-ame-matmul.s

linalg-to-ame-matmul-obj:
	@${BUDDY_OPT} linalg-to-ame-matmul.mlir \
		--lower-linalg-to-ame='target=gem5' \
		--lower-ame \
		--convert-linalg-to-loops \
		--lower-affine \
		--convert-scf-to-cf \
		--expand-strided-metadata \
		--lower-affine \
		--convert-cf-to-llvm \
		--convert-arith-to-llvm \
		--convert-math-to-llvm \
		--convert-func-to-llvm \
		--finalize-memref-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 -target-abi=lp64 \
		-mattr=+m,+xame,+xame-gem5 -verify-machineinstrs \
		-o linalg-to-ame-matmul.o

# linalg.generic -> AME operations -> LLVM IR -> XAME instructions

linalg-to-ame-generic-lower:
	${BUDDY_OPT} linalg-to-ame-generic.mlir \
		--lower-linalg-to-ame='target=gem5' \
		-o linalg-to-ame-generic-lowered.mlir

linalg-to-ame-generic-translate:
	@${BUDDY_OPT} linalg-to-ame-generic.mlir \
		--lower-linalg-to-ame='target=gem5' \
		--lower-ame \
		--convert-linalg-to-loops \
		--lower-affine \
		--convert-scf-to-cf \
		--expand-strided-metadata \
		--lower-affine \
		--convert-cf-to-llvm \
		--convert-arith-to-llvm \
		--convert-math-to-llvm \
		--convert-func-to-llvm \
		--finalize-memref-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir -o linalg-to-ame-generic.ll

linalg-to-ame-generic-asm:
	@${BUDDY_OPT} linalg-to-ame-generic.mlir \
		--lower-linalg-to-ame='target=gem5' \
		--lower-ame \
		--convert-linalg-to-loops \
		--lower-affine \
		--convert-scf-to-cf \
		--expand-strided-metadata \
		--lower-affine \
		--convert-cf-to-llvm \
		--convert-arith-to-llvm \
		--convert-math-to-llvm \
		--convert-func-to-llvm \
		--finalize-memref-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 -target-abi=lp64 \
		-mattr=+m,+xame,+xame-gem5 -verify-machineinstrs \
		-o linalg-to-ame-generic.s

linalg-to-ame-generic-obj:
	@${BUDDY_OPT} linalg-to-ame-generic.mlir \
		--lower-linalg-to-ame='target=gem5' \
		--lower-ame \
		--convert-linalg-to-loops \
		--lower-affine \
		--convert-scf-to-cf \
		--expand-strided-metadata \
		--lower-affine \
		--convert-cf-to-llvm \
		--convert-arith-to-llvm \
		--convert-math-to-llvm \
		--convert-func-to-llvm \
		--finalize-memref-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 -target-abi=lp64 \
		-mattr=+m,+xame,+xame-gem5 -verify-machineinstrs \
		-o linalg-to-ame-generic.o

help:
	@echo "ame-matmul-lower      Lower AME operations to intrinsic calls"
	@echo "ame-matmul-translate  Compile ame-matmul.mlir to LLVM IR (.ll)"
	@echo "ame-matmul-asm        Compile ame-matmul.mlir to XAME assembly (.s; default)"
	@echo "ame-matmul-obj        Compile ame-matmul.mlir to a RISC-V object (.o)"
	@echo "ame-matmul-4x4-lower Lower the 4x4 example to AME intrinsic calls"
	@echo "ame-matmul-4x4-translate Compile the 4x4 example to LLVM IR (.ll)"
	@echo "ame-matmul-4x4-asm   Compile the 4x4 example with the AME_gem5 profile (.s)"
	@echo "ame-matmul-4x4-obj   Compile the 4x4 example with the AME_gem5 profile (.o)"
	@echo "ame-matmul-demo-lower Lower the initialized 4x4 demo to intrinsic calls"
	@echo "ame-matmul-demo-translate Translate the initialized demo to LLVM IR (.ll)"
	@echo "ame-matmul-demo-asm  Compile the initialized demo to XAME assembly (.s)"
	@echo "ame-matmul-demo-obj  Compile the initialized demo to a RISC-V object (.o)"
	@echo "linalg-to-ame-matmul-lower Convert Linalg matmul to AME operations"
	@echo "linalg-to-ame-matmul-translate Compile Linalg matmul to LLVM IR (.ll)"
	@echo "linalg-to-ame-matmul-asm Compile Linalg matmul to XAME assembly (.s)"
	@echo "linalg-to-ame-matmul-obj Compile Linalg matmul to a RISC-V object (.o)"
	@echo "linalg-to-ame-generic-lower Convert linalg.generic regions to AME operations"
	@echo "linalg-to-ame-generic-translate Compile generic regions to LLVM IR (.ll)"
	@echo "linalg-to-ame-generic-asm Compile generic regions to XAME assembly (.s)"
	@echo "linalg-to-ame-generic-obj Compile generic regions to a RISC-V object (.o)"
	@echo "clean                Remove generated files"

clean:
	$(RM) *-lowered.mlir *.ll *.s *.o
