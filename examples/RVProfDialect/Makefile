BUDDY_OPT ?= ../../build/bin/buddy-opt
MLIR_TRANSLATE ?= ../../llvm/build/bin/mlir-translate
LLC ?= ../../llvm/build/bin/llc
CLANG ?= ../../llvm/build/bin/clang
RISCV_CC ?= ../../../../../result/bin/riscv64-unknown-linux-gnu-gcc
RISCV_NM ?= ../../../../../result/bin/riscv64-unknown-linux-gnu-nm

.PHONY: all clean linalg-prof prof-lower e2e-build

all: linalg-prof prof-lower e2e-build

linalg-prof: linalg.mlir
	@$(BUDDY_OPT) linalg.mlir \
		-rvprof-instrument="rvprof-granularity=linalg" \
		> linalg-prof.mlir

prof-lower: linalg-prof
	@$(BUDDY_OPT) linalg-prof.mlir \
		-lower-rvprof \
		> linalg-lowered.mlir

e2e-build: prof-lower RVProfRuntime.c main.c
	@$(BUDDY_OPT) linalg-lowered.mlir \
		-canonicalize \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-expand-strided-metadata \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-convert-cf-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		> linalg-llvm.mlir
	@$(MLIR_TRANSLATE) --mlir-to-llvmir linalg-llvm.mlir > linalg.ll
	@$(LLC) -filetype=obj -mtriple=riscv64 -target-abi=lp64d -mattr=+m,+a,+f,+d,+c linalg.ll -o linalg.o
	@$(RISCV_CC) -c RVProfRuntime.c -o rvprof_rt.o
	@$(RISCV_CC) -c main.c -o main.o
	@$(RISCV_CC) -no-pie rvprof_rt.o main.o linalg.o -o rvprof_e2e.elf

clean:
	rm -f *.o *.ll linalg-prof.mlir linalg-lowered.mlir linalg-llvm.mlir rvprof_e2e.elf rvprof_trace.json *.perfetto
