#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
LLC := ../../llvm/build/bin/llc
OPT := ../../llvm/build/bin/opt
OPT_FLAG := -O3
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so

RISCV_GNU_TOOLCHAIN := ../../thirdparty/build-riscv-gnu-toolchain
RISCV_GNU_TOOLCHAIN_SYSROOT := ../../thirdparty/build-riscv-gnu-toolchain/sysroot
QEMU := ../../thirdparty/qemu/build/riscv64-linux-user/qemu-riscv64
LOCAL_CLANG := ../../thirdparty/build-local-clang/bin/clang
CROSS_LLI := ../../thirdparty/build-cross-clang/bin/lli
CROSS_MLIR_C_RUNNER_UTILS := ../../thirdparty/build-cross-mlir/lib/libmlir_c_runner_utils.so
CROSS_MLIR_RUNNER_UTILS := ../../thirdparty/build-cross-mlir/lib/libmlir_runner_utils.so
CROSS_MLIR_LIB := ../../thirdparty/build-cross-mlir/lib

MLIR_VECTOR_EXAMPLES := ../MLIRVector

.SECONDEXPANSION:
all-run: $$(run-targets)

################################################################################
# RVV Specific Examples
################################################################################

rvv-scalable-translate-128:
	@${BUDDY_OPT} ./rvv-scalable.mlir \
	    -convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll

rvv-scalable-asm-128:
	@${BUDDY_OPT} ./rvv-scalable.mlir \
	    -convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

rvv-scalable-run-128:
	@${BUDDY_OPT} ./rvv-scalable.mlir \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-scalable-aot-128:
	@${BUDDY_OPT} ./rvv-scalable.mlir \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

# Note: this target will trigger an error to show the limitation.
rvv-scalable-run-256-error:
	@${BUDDY_OPT} ./rvv-scalable.mlir \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=256 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-insert-extract-intrinsics-asm:
	@${LLC} ./rvv-insert-extract-intrinsics.ll \
		-mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=256 \
		--filetype=asm -o log.s

rvv-insert-extract-intrinsics-asm-error:
	@${LLC} ./rvv-insert-extract-intrinsics.ll \
		-mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

rvv-c-setvl-translate:
	@${LOCAL_CLANG} -march=rv64gcv --target=riscv64-unknown-linux-gnu \
		--sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT} --gcc-toolchain=${RISCV_GNU_TOOLCHAIN} \
		./rvv-c-setvl.c -fPIC -S -emit-llvm -o log.ll

rvv-c-setvl-asm:
	@${LOCAL_CLANG} -march=rv64gcv --target=riscv64-unknown-linux-gnu \
		--sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT} --gcc-toolchain=${RISCV_GNU_TOOLCHAIN} \
		./rvv-c-setvl.c -fPIC -S -o log.s

run-targets += rvv-c-setvl-run
rvv-c-setvl-run:
	@${LOCAL_CLANG} -march=rv64gcv --target=riscv64-unknown-linux-gnu \
		--sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT} --gcc-toolchain=${RISCV_GNU_TOOLCHAIN} \
		./rvv-c-setvl.c -fPIC -S -emit-llvm -o log.ll
	@${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
		${CROSS_LLI} --entry-function=main --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic log.ll

rvv-setvl-translate:
	@${BUDDY_OPT} ./rvv-setvl.mlir \
		--lower-rvv -convert-vector-to-llvm -convert-arith-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll

rvv-setvl-asm:
	@${BUDDY_OPT} ./rvv-setvl.mlir \
		--lower-rvv -convert-vector-to-llvm -convert-arith-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-setvl-run
rvv-setvl-run:
	@${BUDDY_OPT} ./rvv-setvl.mlir \
		--lower-rvv -convert-vector-to-llvm -convert-arith-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-loop-mask-asm:
	@${MLIR_OPT} ./rvv-loop-mask.mlir \
		-convert-scf-to-cf \
		-convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-loop-mask-run
rvv-loop-mask-run:
	@${MLIR_OPT} ./rvv-loop-mask.mlir \
		-convert-scf-to-cf \
		-convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-vp-intrinsic-lower:
	@${BUDDY_OPT} ./rvv-vp-intrinsic.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts -o log.mlir

rvv-vp-intrinsic-translate:
	@${BUDDY_OPT} ./rvv-vp-intrinsic.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll

rvv-vp-intrinsic-asm:
	@${BUDDY_OPT} ./rvv-vp-intrinsic.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

rvv-vp-intrinsic-run-error:
	@${BUDDY_OPT} ./rvv-vp-intrinsic.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-vp-intrinsic-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-aot-debug:
	@${MLIR_OPT} ./rvv-vp-intrinsic.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		-debug-only=legalize-types > log.txt 2>&1

rvv-vp-intrinsic-sh-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-sh.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-add-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-add.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-and-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-and.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-div-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-div.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-mul-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-mul.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-sub-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-sub.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-fneg-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-fneg.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-ext-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-ext.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-to-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-to.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-trunc-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-trunc.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-rem-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-rem.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-fma-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-fma.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-merge-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-merge.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-select-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-select.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-or-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-or.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-xor-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-xor.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-max-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-max.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-min-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-min.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-memory-aot:
	@${BUDDY_OPT} rvv-vp-intrinsic-memory.mlir \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		--lower-rvv --lower-bud \
		-convert-vector-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-vp-intrinsic-fma-scalable-aot-error:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-fma-scalable.mlir \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

################################################################################
# Reuse MLIR Vector Examples
################################################################################

rvv-load-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-load.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-load-run
rvv-load-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-load.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-broadcast-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-broadcast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-broadcast-run
rvv-broadcast-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-broadcast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-fma-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/rvv-fma.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=256 \
		--filetype=asm -o log.s

run-targets += rvv-fma-run
rvv-fma-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-fma.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-long-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/rvv-long.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-long-run
rvv-long-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-long.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-transpose-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-transpose.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-transpose-run
rvv-transpose-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-transpose.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-shape-cast-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-shape-cast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-shape-cast-run
rvv-shape-cast-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-shape-cast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-bitcast-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-bitcast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-bitcast-run
rvv-bitcast-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-bitcast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-shuffle-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-shuffle.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-shuffle-run
rvv-shuffle-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-shuffle.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-splat-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-splat.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-splat-run
rvv-splat-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-splat.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-insert-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-insert.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-insert-run
rvv-insert-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-insert.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-reduction-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-reduction.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-reduction-run
rvv-reduction-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-reduction.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-outerproduct-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-outerproduct.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-outerproduct-run
rvv-outerproduct-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-outerproduct.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-createmask-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-createmask.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-createmask-run
rvv-createmask-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-createmask.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-extract-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-extract.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-extract-run
rvv-extract-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-extract.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-maskedload-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-maskedload.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-maskedload-run
rvv-maskedload-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-maskedload.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-maskedstore-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-maskedstore.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-maskedstore-run
rvv-maskedstore-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-maskedstore.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-extract-strided-slice-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-extract-strided-slice.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-extract-strided-slice-run
rvv-extract-strided-slice-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-extract-strided-slice.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-constant-mask-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-constant-mask.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-constant-mask-run
rvv-constant-mask-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-constant-mask.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-expand-load-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-expand-load.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-expand-load-run
rvv-expand-load-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-expand-load.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-compressstore-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-compressstore.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-compressstore-run
rvv-compressstore-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-compressstore.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-insert-strided-slice-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-insert-strided-slice.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-insert-strided-slice-run
rvv-insert-strided-slice-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-insert-strided-slice.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-scatter-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-scatter.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-scatter-run
rvv-scatter-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-scatter.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}
