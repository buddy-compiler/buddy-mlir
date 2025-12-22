#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
BUDDY_LLC := ../../build/bin/buddy-llc

.PHONY: all clean vmadot vmadotu vmadotsu vmadotus vfmadot help

all: vmadot vmadotu vmadotsu vmadotus vfmadot


vmadot: vmadot.mlir
	$(BUDDY_OPT) $< -o vmadot-out.mlir

vmadotu: vmadotu.mlir
	$(BUDDY_OPT) $< -o vmadotu-out.mlir

vmadotsu: vmadotsu.mlir
	$(BUDDY_OPT) $< -o vmadotsu-out.mlir

vmadotus: vmadotus.mlir
	$(BUDDY_OPT) $< -o vmadotus-out.mlir

vfmadot: vfmadot.mlir
	$(BUDDY_OPT) $< -o vfmadot-out.mlir

clean:
	rm -f *-out.mlir


vmadot-lower:
	@${BUDDY_OPT} ./vmadot.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadot-translate:
	@${BUDDY_OPT} ./vmadot.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadot-asm:
	@${BUDDY_OPT} ./vmadot.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v \
		-o log.s


vmadotu-lower:
	@${BUDDY_OPT} ./vmadotu.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadotu-translate:
	@${BUDDY_OPT} ./vmadotu.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadotu-asm:
	@${BUDDY_OPT} ./vmadotu.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v \
		-o log.s


vmadotsu-lower:
	@${BUDDY_OPT} ./vmadotsu.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadotsu-translate:
	@${BUDDY_OPT} ./vmadotsu.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadotsu-asm:
	@${BUDDY_OPT} ./vmadotsu.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v \
		-o log.s


vmadotus-lower:
	@${BUDDY_OPT} ./vmadotus.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadotus-translate:
	@${BUDDY_OPT} ./vmadotus.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadotus-asm:
	@${BUDDY_OPT} ./vmadotus.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v \
		-o log.s


vfmadot-lower:
	@${BUDDY_OPT} ./vfmadot.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vfmadot-translate:
	@${BUDDY_OPT} ./vfmadot.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vfmadot-asm:
	@${BUDDY_OPT} ./vfmadot.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v \
		-o log.s
