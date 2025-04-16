#!/bin/bash

# Build Directories
MLIR_BUILD_DIR := ../../llvm/build/
BUDDY_MLIR_BUILD_DIR := ../../build/
CROSS_BUDDY_MLIR_BUILD_DIR := ../../build-cross-rv/
CROSS_LLVM_BUILD_DIR := ../../llvm/build-cross-clang-rv/
CROSS_MLIR_BUILD_DIR := ../../llvm/build-cross-mlir-rv/

# Buddy MLIR Tools
BUDDY_OPT := ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
BUDDY_TRANSLATE := ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate

# Core LLVM/MLIR Tools
MLIR_OPT := ${MLIR_BUILD_DIR}/bin/mlir-opt
MLIR_TRANSLATE := ${MLIR_BUILD_DIR}/bin/mlir-translate
MLIR_CPU_RUNNER := ${MLIR_BUILD_DIR}/bin/mlir-runner
LLC := ${MLIR_BUILD_DIR}/bin/llc
LLI := ${MLIR_BUILD_DIR}/bin/lli
OPT := ${MLIR_BUILD_DIR}/bin/opt
LOCAL_CLANG := ${MLIR_BUILD_DIR}/bin/clang

# RISC-V GNU Toolchain
RISCV_GNU_TOOLCHAIN := ${BUDDY_MLIR_BUILD_DIR}/thirdparty/riscv-gnu-toolchain
RISCV_GNU_TOOLCHAIN_SYSROOT := ${RISCV_GNU_TOOLCHAIN}/sysroot
QEMU := ${RISCV_GNU_TOOLCHAIN}/bin/qemu-riscv64

# Cross Compiled Toolchain
CROSS_BUDDY_MLIR_LIB := ${CROSS_BUDDY_MLIR_BUILD_DIR}/lib/
CROSS_LLI := ${CROSS_LLVM_BUILD_DIR}/bin/lli
CROSS_MLIR_CPU_RUNNER := ${CROSS_MLIR_BUILD_DIR}/bin/mlir-runner
CROSS_MLIR_C_RUNNER_UTILS := ${CROSS_MLIR_BUILD_DIR}/lib/libmlir_c_runner_utils.so
CROSS_MLIR_RUNNER_UTILS := ${CROSS_MLIR_BUILD_DIR}/lib/libmlir_runner_utils.so
CROSS_MLIR_LIB := ${CROSS_MLIR_BUILD_DIR}/lib

# Optimization Flag
OPT_FLAG := -O3

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ${MLIR_BUILD_DIR}/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ${MLIR_BUILD_DIR}//lib/libmlir_c_runner_utils.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ${MLIR_BUILD_DIR}/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ${MLIR_BUILD_DIR}/lib/libmlir_c_runner_utils.dylib
MTRIPLE := x86_64-apple-darwin
endif

MLIR_VECTOR_EXAMPLES := ../MLIRVector

.SECONDEXPANSION:
all-run: $$(run-targets)

################################################################################
# RVV Specific Examples
################################################################################

rvv-scalable-translate-128:
	@${BUDDY_OPT} ./rvv-scalable.mlir \
	    -convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll

rvv-scalable-asm-128:
	@${BUDDY_OPT} ./rvv-scalable.mlir \
	    -convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

rvv-scalable-run-128:
	@${BUDDY_OPT} ./rvv-scalable.mlir \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-scalable-aot-128:
	@${BUDDY_OPT} ./rvv-scalable.mlir \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-scalable-run-256:
	@${BUDDY_OPT} ./rvv-scalable.mlir \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-insert-extract-intrinsics-asm:
	@${LLC} ./rvv-insert-extract-intrinsics.ll \
		-mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=256 \
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
	@${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
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
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-vscale-asm:
	@${BUDDY_OPT} ./rvv-vscale.mlir \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=256 \
		--filetype=asm -o log.s

rvv-vscale-128-run:
	@${BUDDY_OPT} ./rvv-vscale.mlir \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-vscale-256-run:
	@${BUDDY_OPT} ./rvv-vscale.mlir \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-vscale-512-run:
	@${BUDDY_OPT} ./rvv-vscale.mlir \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-vscale-128-aot:
	@${BUDDY_OPT} ./rvv-vscale.mlir \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64  -mattr=+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o  \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vscale-256-aot:
	@${BUDDY_OPT} ./rvv-vscale.mlir \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64  -mattr=+v -riscv-v-vector-bits-min=256 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o  \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vscale-512-aot:
	@${BUDDY_OPT} ./rvv-vscale.mlir \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64  -mattr=+v -riscv-v-vector-bits-min=512 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o  \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-loop-mask-asm:
	@${MLIR_OPT} ./rvv-loop-mask.mlir \
		-convert-scf-to-cf \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-loop-mask-run
rvv-loop-mask-run:
	@${MLIR_OPT} ./rvv-loop-mask.mlir \
		-convert-scf-to-cf \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-vp-intrinsic-lower:
	@${BUDDY_OPT} ./rvv-vp-intrinsic.mlir \
		--convert-scf-to-cf \
		--lower-rvv --lower-bud --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts -o log.mlir

rvv-vp-intrinsic-translate:
	@${BUDDY_OPT} ./rvv-vp-intrinsic.mlir \
		-lower-vector-exp \
		--lower-rvv \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll

rvv-vp-intrinsic-asm:
	@${BUDDY_OPT} ./rvv-vp-intrinsic.mlir \
		-lower-vector-exp \
		--lower-rvv \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

rvv-vp-intrinsic-run:
	@${BUDDY_OPT} ./rvv-vp-intrinsic.mlir \
		-lower-vector-exp \
		--lower-rvv \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-vp-intrinsic-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic.mlir \
		-lower-vector-exp \
		--lower-rvv \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-sh-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-sh.mlir \
		--lower-rvv \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64  -mattr=+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o  \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-sh-jit:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-sh.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+v \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-vp-intrinsic-add-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-add.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-add-jit:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-add.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-vp-intrinsic-and-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-and.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-and-jit:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-and.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-vp-intrinsic-div-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-div.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-mul-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-mul.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-sub-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-sub.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-fneg-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-fneg.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-ext-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-ext.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-to-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-to.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-trunc-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-trunc.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-rem-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-rem.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-fma-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-fma.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-merge-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-merge.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-select-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-select.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-or-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-or.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-xor-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-xor.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-max-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-max.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-min-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-min.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

# TODO: Fix Me
rvv-vp-intrinsic-memory-aot:
	@${BUDDY_OPT} rvv-vp-intrinsic-memory.mlir \
		-convert-vector-to-scf \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		--lower-rvv \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

# TODO: Fix Me
rvv-vp-intrinsic-memory-scalable-aot:
	@${BUDDY_OPT} rvv-vp-intrinsic-memory-scalable.mlir \
		-convert-linalg-to-loops -lower-affine \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--lower-rvv \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-fma-scalable-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-fma-scalable.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-fneg-scalable-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-fneg-scalable.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-sh-scalable-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-sh-scalable.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-add-scalable-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-add-scalable.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-and-scalable-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-and-scalable.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-div-scalable-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-div-scalable.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-ext-scalable-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-ext-scalable.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-max-scalable-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-max-scalable.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-merge-scalable-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-merge-scalable.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-min-scalable-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-min-scalable.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-mul-scalable-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-mul-scalable.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-or-scalable-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-or-scalable.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-rem-scalable-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-rem-scalable.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-select-scalable-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-select-scalable.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-sub-scalable-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-sub-scalable.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-to-scalable-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-to-scalable.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-trunc-scalable-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-trunc-scalable.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-xor-scalable-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-xor-scalable.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-rem-error:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-rem-error.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-fmul-reduce-error:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-fmul-reduce-error.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-vp-intrinsic-mul-reduce-aot:
	@${BUDDY_OPT} ./rvv-vp-intrinsic-mul-reduce.mlir \
		-convert-vector-to-scf \
		--convert-scf-to-cf \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
		--convert-func-to-llvm --reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

################################################################################
# Reuse MLIR Vector Examples
################################################################################

rvv-load-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-load.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-load-run
rvv-load-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-load.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-broadcast-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-broadcast.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-broadcast-run
rvv-broadcast-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-broadcast.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-fma-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-fma.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=256 \
		--filetype=asm -o log.s

run-targets += rvv-fma-run
rvv-fma-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-fma.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-long-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-long.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-long-run
rvv-long-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-long.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-transpose-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-transpose.mlir \
		-convert-vector-to-scf -convert-scf-to-cf \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-transpose-run
rvv-transpose-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-transpose.mlir \
		-convert-vector-to-scf -convert-scf-to-cf \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-shape-cast-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-shape-cast.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-shape-cast-run
rvv-shape-cast-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-shape-cast.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-bitcast-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-bitcast.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-bitcast-run
rvv-bitcast-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-bitcast.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-shuffle-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-shuffle.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-shuffle-run
rvv-shuffle-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-shuffle.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-splat-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-splat.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-splat-run
rvv-splat-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-splat.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-insert-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-insert.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-insert-run
rvv-insert-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-insert.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-reduction-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-reduction.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-reduction-run
rvv-reduction-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-reduction.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-outerproduct-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-outerproduct.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-outerproduct-run
rvv-outerproduct-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-outerproduct.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-createmask-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-createmask.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-create-mask-run
rvv-create-mask-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-create-mask.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-extract-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-extract.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-extract-run
rvv-extract-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-extract.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-maskedload-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-maskedload.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-maskedload-run
rvv-maskedload-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-maskedload.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-maskedstore-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-maskedstore.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-maskedstore-run
rvv-maskedstore-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-maskedstore.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-extract-strided-slice-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-extract-strided-slice.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-extract-strided-slice-run
rvv-extract-strided-slice-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-extract-strided-slice.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-constant-mask-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-constant-mask.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-constant-mask-run
rvv-constant-mask-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-constant-mask.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-expand-load-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-expandload.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-expand-load-run
rvv-expand-load-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-expandload.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-compressstore-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-compressstore.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-compressstore-run
rvv-compressstore-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-compressstore.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-insert-strided-slice-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-insert-strided-slice.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-insert-strided-slice-run
rvv-insert-strided-slice-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-insert-strided-slice.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-scatter-asm:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-scatter.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-scatter-run
rvv-scatter-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-scatter.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS} --dlopen=${CROSS_MLIR_RUNNER_UTILS}
