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
OPT_FLAG := -O0

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ${MLIR_BUILD_DIR}/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ${MLIR_BUILD_DIR}//lib/libmlir_c_runner_utils.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ${MLIR_BUILD_DIR}/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ${MLIR_BUILD_DIR}/lib/libmlir_c_runner_utils.dylib
MTRIPLE := x86_64-apple-darwin
endif

rvv-setvl-lower:
	@${BUDDY_OPT} ./rvv-setvl.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-convert-arith-to-llvm \
		-lower-rvv \
		-convert-vector-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts \
		-o ./log.mlir

rvv-setvl-translate:
	@${BUDDY_OPT} ./rvv-setvl.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-convert-arith-to-llvm \
		-lower-rvv \
		-convert-vector-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir -o log.ll

rvv-setvl-128-run:
	@${BUDDY_OPT} ./rvv-setvl.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-convert-arith-to-llvm \
		-lower-rvv \
		-convert-vector-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} -march=riscv64 -mattr=+m,+d,+v,-zcf \
		-dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-setvl-256-run:
	@${BUDDY_OPT} ./rvv-setvl.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-convert-arith-to-llvm \
		-lower-rvv \
		-convert-vector-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} -march=riscv64 -mattr=+m,+d,+v,-zcf \
		-dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-rsqrt-lower:
	@${BUDDY_OPT} ./rvv-rsqrt.mlir \
		--convert-scf-to-cf \
		--convert-math-to-llvm \
		-convert-arith-to-llvm \
		-convert-cf-to-llvm \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts \
		-o ./log.mlir

rvv-rsqrt-translate:
	@${BUDDY_OPT} ./rvv-rsqrt.mlir \
		--convert-scf-to-cf \
		--convert-math-to-llvm \
		-convert-arith-to-llvm \
		-convert-cf-to-llvm \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
		${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll

rvv-rsqrt-aot:
	@${BUDDY_OPT} ./rvv-rsqrt.mlir \
		-convert-scf-to-cf \
		--convert-math-to-llvm \
		-convert-arith-to-llvm \
		-convert-cf-to-llvm \
		--lower-rvv -convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm\
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

rvv-mul-add-lower:
	@${BUDDY_OPT} ./rvv-mul-add.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-convert-arith-to-llvm \
		-convert-cf-to-llvm \
		-lower-rvv \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts \
		-o ./log.mlir

rvv-mul-add-translate:
	@${BUDDY_OPT} ./rvv-mul-add.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-convert-arith-to-llvm \
		-convert-cf-to-llvm \
		-lower-rvv \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir -o log.ll

rvv-mul-add-run:
	@${BUDDY_OPT} ./rvv-mul-add.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-convert-arith-to-llvm \
		-convert-cf-to-llvm \
		-lower-rvv \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} \
			-cpu max \
	${CROSS_LLI} -march=riscv64 -mattr=+m,+d,+v,-zcf \
		-dlopen=${CROSS_MLIR_C_RUNNER_UTILS} \
		-dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-stripmining-lower:
	@${BUDDY_OPT} ./rvv-stripmining.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-rvv \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-convert-arith-to-llvm \
		-convert-cf-to-llvm \
		-reconcile-unrealized-casts \
		-o ./log.mlir

rvv-stripmining-translate:
	@${BUDDY_OPT} ./rvv-stripmining.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-rvv \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-convert-arith-to-llvm \
		-convert-cf-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir -o log.ll

rvv-stripmining-run:
	@${BUDDY_OPT} ./rvv-stripmining.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-rvv \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-convert-arith-to-llvm \
		-convert-cf-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} \
			-cpu max \
	${CROSS_LLI} -march=riscv64 -mattr=+m,+d,+v,-zcf \
		-dlopen=${CROSS_MLIR_C_RUNNER_UTILS} \
		-dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-stripmining-aot:
	@${BUDDY_OPT} ./rvv-stripmining.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-rvv \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-convert-arith-to-llvm \
		-convert-cf-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o \
		-mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out
