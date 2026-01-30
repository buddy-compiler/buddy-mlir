#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
BUDDY_LLC := ../../build/bin/buddy-llc

.PHONY: all clean vmadot vmadotu vmadotsu vmadotus vfmadot help \
        vmadot-lower vmadot-translate vmadot-asm vmadot-run \
        vmadotu-lower vmadotu-translate vmadotu-asm vmadotu-run \
        vmadotsu-lower vmadotsu-translate vmadotsu-asm vmadotsu-run \
        vmadotus-lower vmadotus-translate vmadotus-asm vmadotus-run \
        vfmadot-lower vfmadot-translate vfmadot-asm vfmadot-run \
        vmadotn vmadotnu vmadotnsu vmadotnus vfmadotn \
        vmadotn-lower vmadotn-translate vmadotn-asm vmadotn-run \
        vmadotnu-lower vmadotnu-translate vmadotnu-asm vmadotnu-run \
        vmadotnsu-lower vmadotnsu-translate vmadotnsu-asm vmadotnsu-run \
        vmadotnus-lower vmadotnus-translate vmadotnus-asm vmadotnus-run \
        vfmadotn-lower vfmadotn-translate vfmadotn-asm vfmadotn-run \
        vmadot1-lower vmadot1-translate vmadot1-asm vmadot1-run \
        vmadot1u-lower vmadot1u-translate vmadot1u-asm \
        vmadot1su-lower vmadot1su-translate vmadot1su-asm \
        vmadot1us-lower vmadot1us-translate vmadot1us-asm \
        vmadot2-lower vmadot2-translate vmadot2-asm vmadot2-run \
        vmadot2u-lower vmadot2u-translate vmadot2u-asm \
        vmadot2su-lower vmadot2su-translate vmadot2su-asm \
        vmadot2us-lower vmadot2us-translate vmadot2us-asm \
        vmadot3-lower vmadot3-translate vmadot3-asm vmadot3-run \
        vmadot3u-lower vmadot3u-translate vmadot3u-asm \
        vmadot3su-lower vmadot3su-translate vmadot3su-asm \
        vmadot3us-lower vmadot3us-translate vmadot3us-asm \
        vfmadot1-lower vfmadot1-translate vfmadot1-asm vfmadot1-run \
        vfmadot2-lower vfmadot2-translate vfmadot2-asm vfmadot2-run \
        vfmadot3-lower vfmadot3-translate vfmadot3-asm vfmadot3-run

all: vmadot vmadotu vmadotsu vmadotus vfmadot vmadotn vmadotnu vmadotnsu vmadotnus vfmadotn


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

vmadotn: vmadotn.mlir
	$(BUDDY_OPT) $< -o vmadotn-out.mlir

vmadotnu: vmadotnu.mlir
	$(BUDDY_OPT) $< -o vmadotnu-out.mlir

vmadotnsu: vmadotnsu.mlir
	$(BUDDY_OPT) $< -o vmadotnsu-out.mlir

vmadotnus: vmadotnus.mlir
	$(BUDDY_OPT) $< -o vmadotnus-out.mlir

vfmadotn: vfmadotn.mlir
	$(BUDDY_OPT) $< -o vfmadotn-out.mlir

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
		-mattr=+m,+v,+zvfh,+buddyext \
		-o log.s

vfmadot-run:
	@${BUDDY_OPT} ./vfmadot_print_test.mlir \
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
		-mattr=+m,+v,+zvfh,+buddyext \
		-o vfmadot.s
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vfmadot.s runtime_vfmadot.c -o vfmadot_test


vmadot1-lower:
	@${BUDDY_OPT} ./vmadot1.mlir \
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

vmadot1-translate:
	@${BUDDY_OPT} ./vmadot1.mlir \
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

vmadot1-asm:
	@${BUDDY_OPT} ./vmadot1.mlir \
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

vmadot2-lower:
	@${BUDDY_OPT} ./vmadot2.mlir \
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

vmadot2-translate:
	@${BUDDY_OPT} ./vmadot2.mlir \
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

vmadot2-asm:
	@${BUDDY_OPT} ./vmadot2.mlir \
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

vmadot3-lower:
	@${BUDDY_OPT} ./vmadot3.mlir \
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

vmadot3-translate:
	@${BUDDY_OPT} ./vmadot3.mlir \
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

vmadot3-asm:
	@${BUDDY_OPT} ./vmadot3.mlir \
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

vmadot1-run:
	@${BUDDY_OPT} ./vmadot1_print_test.mlir \
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
		-o vmadot1.s
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vmadot1.s runtime_vmadot1.c -o vmadot1_test

vmadot2-run:
	@${BUDDY_OPT} ./vmadot2_print_test.mlir \
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
		-o vmadot2.s
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vmadot2.s runtime_vmadot2.c -o vmadot2_test

vmadot3-run:
	@${BUDDY_OPT} ./vmadot3_print_test.mlir \
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
		-o vmadot3.s
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vmadot3.s runtime_vmadot3.c -o vmadot3_test

vmadot1u-lower:
	@${BUDDY_OPT} ./vmadot1u.mlir \
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

vmadot1u-translate:
	@${BUDDY_OPT} ./vmadot1u.mlir \
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

vmadot1u-asm:
	@${BUDDY_OPT} ./vmadot1u.mlir \
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

vmadot1su-lower:
	@${BUDDY_OPT} ./vmadot1su.mlir \
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

vmadot1su-translate:
	@${BUDDY_OPT} ./vmadot1su.mlir \
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

vmadot1su-asm:
	@${BUDDY_OPT} ./vmadot1su.mlir \
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

vmadot1us-lower:
	@${BUDDY_OPT} ./vmadot1us.mlir \
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

vmadot1us-translate:
	@${BUDDY_OPT} ./vmadot1us.mlir \
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

vmadot1us-asm:
	@${BUDDY_OPT} ./vmadot1us.mlir \
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

vmadot2u-lower:
	@${BUDDY_OPT} ./vmadot2u.mlir \
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

vmadot2u-translate:
	@${BUDDY_OPT} ./vmadot2u.mlir \
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

vmadot2u-asm:
	@${BUDDY_OPT} ./vmadot2u.mlir \
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

vmadot2su-lower:
	@${BUDDY_OPT} ./vmadot2su.mlir \
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

vmadot2su-translate:
	@${BUDDY_OPT} ./vmadot2su.mlir \
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

vmadot2su-asm:
	@${BUDDY_OPT} ./vmadot2su.mlir \
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

vmadot2us-lower:
	@${BUDDY_OPT} ./vmadot2us.mlir \
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

vmadot2us-translate:
	@${BUDDY_OPT} ./vmadot2us.mlir \
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

vmadot2us-asm:
	@${BUDDY_OPT} ./vmadot2us.mlir \
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

vmadot3u-lower:
	@${BUDDY_OPT} ./vmadot3u.mlir \
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

vmadot3u-translate:
	@${BUDDY_OPT} ./vmadot3u.mlir \
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

vmadot3u-asm:
	@${BUDDY_OPT} ./vmadot3u.mlir \
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

vmadot3su-lower:
	@${BUDDY_OPT} ./vmadot3su.mlir \
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

vmadot3su-translate:
	@${BUDDY_OPT} ./vmadot3su.mlir \
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

vmadot3su-asm:
	@${BUDDY_OPT} ./vmadot3su.mlir \
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

vmadot3us-lower:
	@${BUDDY_OPT} ./vmadot3us.mlir \
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

vmadot3us-translate:
	@${BUDDY_OPT} ./vmadot3us.mlir \
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

vmadot3us-asm:
	@${BUDDY_OPT} ./vmadot3us.mlir \
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

vmadotn-lower:
	@${BUDDY_OPT} ./vmadotn.mlir \
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

vmadotn-translate:
	@${BUDDY_OPT} ./vmadotn.mlir \
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

vmadotn-asm:
	@${BUDDY_OPT} ./vmadotn.mlir \
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

vmadotn-run:
	@${BUDDY_OPT} ./vmadotn_print_test.mlir \
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
		-o vmadotn.s
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vmadotn.s runtime_vmadotn.c -o vmadotn_test


vmadotnu-lower:
	@${BUDDY_OPT} ./vmadotnu.mlir \
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

vmadotnu-translate:
	@${BUDDY_OPT} ./vmadotnu.mlir \
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

vmadotnu-asm:
	@${BUDDY_OPT} ./vmadotnu.mlir \
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

vmadotnu-run:
	@${BUDDY_OPT} ./vmadotnu_print_test.mlir \
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
		-o vmadotnu.s
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vmadotnu.s runtime_vmadotnu.c -o vmadotnu_test


vmadotnsu-lower:
	@${BUDDY_OPT} ./vmadotnsu.mlir \
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

vmadotnsu-translate:
	@${BUDDY_OPT} ./vmadotnsu.mlir \
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

vmadotnsu-asm:
	@${BUDDY_OPT} ./vmadotnsu.mlir \
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

vmadotnsu-run:
	@${BUDDY_OPT} ./vmadotnsu_print_test.mlir \
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
		-o vmadotnsu.s
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vmadotnsu.s runtime_vmadotnsu.c -o vmadotnsu_test


vmadotnus-lower:
	@${BUDDY_OPT} ./vmadotnus.mlir \
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

vmadotnus-translate:
	@${BUDDY_OPT} ./vmadotnus.mlir \
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

vmadotnus-asm:
	@${BUDDY_OPT} ./vmadotnus.mlir \
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

vmadotnus-run:
	@${BUDDY_OPT} ./vmadotnus_print_test.mlir \
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
		-o vmadotnus.s
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vmadotnus.s runtime_vmadotnus.c -o vmadotnus_test


vfmadot1-lower:
	@${BUDDY_OPT} ./vfmadot1.mlir \
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

vfmadot1-translate:
	@${BUDDY_OPT} ./vfmadot1.mlir \
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

vfmadot1-asm:
	@${BUDDY_OPT} ./vfmadot1.mlir \
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
		-mattr=+m,+v,+zvfh,+buddyext \
		-o log.s

vfmadot2-lower:
	@${BUDDY_OPT} ./vfmadot2.mlir \
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

vfmadot2-translate:
	@${BUDDY_OPT} ./vfmadot2.mlir \
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

vfmadot2-asm:
	@${BUDDY_OPT} ./vfmadot2.mlir \
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
		-mattr=+m,+v,+zvfh,+buddyext \
		-o log.s

vfmadot3-lower:
	@${BUDDY_OPT} ./vfmadot3.mlir \
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

vfmadot3-translate:
	@${BUDDY_OPT} ./vfmadot3.mlir \
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

vfmadot3-asm:
	@${BUDDY_OPT} ./vfmadot3.mlir \
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
		-mattr=+m,+v,+zvfh,+buddyext \
		-o log.s

vfmadot1-run:
	@${BUDDY_OPT} ./vfmadot1_print_test.mlir \
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
		-mattr=+m,+v,+zvfh,+buddyext \
		-o vfmadot1.s
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vfmadot1.s runtime_vfmadot1.c -o vfmadot1_test

vfmadot2-run:
	@${BUDDY_OPT} ./vfmadot2_print_test.mlir \
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
		-mattr=+m,+v,+zvfh,+buddyext \
		-o vfmadot2.s
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vfmadot2.s runtime_vfmadot2.c -o vfmadot2_test

vfmadot3-run:
	@${BUDDY_OPT} ./vfmadot3_print_test.mlir \
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
		-mattr=+m,+v,+zvfh,+buddyext \
		-o vfmadot3.s
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vfmadot3.s runtime_vfmadot3.c -o vfmadot3_test

vfmadotn-lower:
	@${BUDDY_OPT} ./vfmadotn.mlir \
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

vfmadotn-translate:
	@${BUDDY_OPT} ./vfmadotn.mlir \
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

vfmadotn-asm:
	@${BUDDY_OPT} ./vfmadotn.mlir \
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
		-mattr=+m,+v,+zvfh,+buddyext \
		-o log.s

vfmadotn-run:
	@${BUDDY_OPT} ./vfmadotn_print_test.mlir \
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
		-mattr=+m,+v,+zvfh,+buddyext \
		-o vfmadotn.s
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vfmadotn.s runtime_vfmadotn.c -o vfmadotn_test

