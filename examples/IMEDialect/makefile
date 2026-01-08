#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
BUDDY_LLC := ../../build/bin/buddy-llc

.PHONY: all clean vmadot vmadotu vmadotsu vmadotus vfmadot help \
        vmadot-lower vmadot-translate vmadot-asm vmadot-run \
        vmadotu-lower vmadotu-translate vmadotu-asm vmadotu-run \
        vmadotsu-lower vmadotsu-translate vmadotsu-asm vmadotsu-run \
        vmadotus-lower vmadotus-translate vmadotus-asm vmadotus-run \
        vfmadot-lower vfmadot-translate vfmadot-asm

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
		-mattr=+m,+v,+buddyext \
		-o log.s

vmadot-run:
	@${BUDDY_OPT} ./vmadot_print_test.mlir \
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
		-mattr=+m,+v,+buddyext \
		-o vmadot.s
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vmadot.s runtime_vmadot.c -o vmadot_test


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
		-mattr=+m,+v,+buddyext \
		-o log.s

vmadotu-run:
	@${BUDDY_OPT} ./vmadotu_print_test.mlir \
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
		-mattr=+m,+v,+buddyext \
		-o vmadotu.s
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vmadotu.s runtime_vmadotu.c -o vmadotu_test

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
		-mattr=+m,+v,+buddyext \
		-o log.s

vmadotsu-run:
	@${BUDDY_OPT} ./vmadotsu_print_test.mlir \
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
		-mattr=+m,+v,+buddyext \
		-o vmadotsu.s
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vmadotsu.s runtime_vmadotsu.c -o vmadotsu_test

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
		-mattr=+m,+v,+buddyext \
		-o log.s

vmadotus-run:
	@${BUDDY_OPT} ./vmadotus_print_test.mlir \
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
		-mattr=+m,+v,+buddyext \
		-o vmadotus.s
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vmadotus.s runtime_vmadotus.c -o vmadotus_test

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
		-mattr=+m,+v,+buddyext \
		-o log.s
