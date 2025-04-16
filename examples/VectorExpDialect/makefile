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

vector-exp-load-dynamic-lower:
	@${BUDDY_OPT} ./vector-exp-load-dynamic.mlir \
		-o ./log.mlir

vector-exp-load-original-lower:
	@${BUDDY_OPT} ./vector-exp-load-original.mlir \
		-convert-vector-to-llvm \
		-o ./log.mlir

vector-exp-load-original-translate:
	@${BUDDY_OPT} ./vector-exp-load-original.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll

vector-exp-load-original-asm:
	@${BUDDY_OPT} ./vector-exp-load-original.mlir \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

vector-exp-config-lower:
	@${BUDDY_OPT} ./vector-exp-predication.mlir \
		-lower-vector-exp \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts \
		-o ./log.mlir

vector-exp-config-translate:
	@${BUDDY_OPT} ./vector-exp-predication.mlir \
		-lower-vector-exp \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll

vector-exp-config-run:
	@${BUDDY_OPT} ./vector-exp-predication.mlir \
		-lower-vector-exp \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vector-exp-predication-memory-lower:
	@${BUDDY_OPT} ./vector-exp-predication-memory.mlir \
		-lower-vector-exp \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts \
		-o ./log.mlir

vector-exp-predication-memory-run:
	@${BUDDY_OPT} ./vector-exp-predication-memory.mlir \
		-lower-vector-exp \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts |\
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} -march=riscv64 -mattr=+m,+d,+v \
		-dlopen=${CROSS_MLIR_C_RUNNER_UTILS} \
		-dlopen=${CROSS_MLIR_RUNNER_UTILS}

vector-exp-predication-matmul-run:
	@${BUDDY_OPT} ./vector-exp-predication-matmul.mlir \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-vector-exp \
		-lower-rvv \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts |\
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} -march=riscv64 -mattr=+m,+d,+v \
		-dlopen=${CROSS_MLIR_C_RUNNER_UTILS} \
		-dlopen=${CROSS_MLIR_RUNNER_UTILS}

vector-exp-predication-matmul-aot:
	@${BUDDY_OPT} ./vector-exp-predication-matmul.mlir \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-vector-exp \
		-lower-rvv \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts |\
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64  -mattr=+v,+m -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o  \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

vector-exp-predication-matmul-elf:
	@${BUDDY_OPT} ./vector-exp-predication-matmul.mlir \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-vector-exp \
		-lower-rvv \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts |\
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${LLC} -mtriple riscv64  -mattr=+v,+m -riscv-v-vector-bits-min=128 -filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o  \
		-static \
		-L${CROSS_BUDDY_MLIR_LIB} \
		-lStaticMLIRCRunnerUtils -lStaticMLIRRunnerUtils \
		-lstdc++ \
		-o matmul-predication-perf.elf

vector-exp-scalar-matmul-elf:
	@${BUDDY_OPT} ./vector-exp-scalar-matmul.mlir \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
        -convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
        -convert-func-to-llvm \
		-reconcile-unrealized-casts |\
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${LLC} -O0 -mtriple riscv64  -mattr=+v,+m -riscv-v-vector-bits-min=128 -filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o  \
		-static \
		-L${CROSS_BUDDY_MLIR_LIB} \
		-lStaticMLIRCRunnerUtils -lStaticMLIRRunnerUtils \
		-lstdc++ \
		-o matmul-scalar-perf.elf

vector-exp-O3-matmul-elf:
	@${BUDDY_OPT} ./vector-exp-scalar-matmul.mlir \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
        -convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
        -convert-func-to-llvm \
		-reconcile-unrealized-casts |\
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${LLC} -O3 -mtriple riscv64  -mattr=+v,+m -riscv-v-vector-bits-min=128 -filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o  \
		-static \
		-L${CROSS_BUDDY_MLIR_LIB} \
		-lStaticMLIRCRunnerUtils -lStaticMLIRRunnerUtils \
		-lstdc++ \
		-o matmul-o3-perf.elf

vector-exp-add-mask-run:
	@${BUDDY_OPT} ./vector-exp-add-mask.mlir \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} \
			-cpu max \
	${CROSS_LLI} -march=riscv64 -mattr=+m,+d,+v \
		-dlopen=${CROSS_MLIR_C_RUNNER_UTILS} \
		-dlopen=${CROSS_MLIR_RUNNER_UTILS}

vector-exp-add-mask-asm:
	@${BUDDY_OPT} ./vector-exp-add-mask.mlir \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

vector-exp-add-predication-run:
	@${BUDDY_OPT} ./vector-exp-add-predication.mlir \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-vector-exp \
		-lower-rvv \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max \
	${CROSS_LLI} -march=riscv64 -mattr=+m,+d,+v \
		-dlopen=${CROSS_MLIR_C_RUNNER_UTILS} \
		-dlopen=${CROSS_MLIR_RUNNER_UTILS}

vector-exp-add-predication-asm:
	@${BUDDY_OPT} ./vector-exp-add-predication.mlir \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-vector-exp \
		-lower-rvv \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

vector-exp-dynamic-vector-dump:
	@${BUDDY_OPT} ./vector-exp-dynamic-vector.mlir \
		-o log.mlir

vector-exp-dynamic-vector-lower:
	@${BUDDY_OPT} ./vector-exp-dynamic-vector.mlir \
		-lower-vector-exp \
		-o log.mlir

vector-exp-dynamic-vector-run:
	@${BUDDY_OPT} ./vector-exp-dynamic-vector.mlir \
		-lower-vector-exp \
		-convert-vector-to-llvm \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-rvv \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64  -mattr=+v,+m -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o  \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max a.out

vector-exp-iteration-aot:
	@${BUDDY_OPT} ./vector-exp-iteration.mlir \
		-lower-vector-exp \
		-lower-affine \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-vector-to-llvm \
		-convert-index-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir -o log.ll
	${LOCAL_CLANG} -O3 log.ll \
		-march=rv64gcv --target=riscv64-unknown-linux-gnu -fPIC \
		--sysroot=${RISCV_GNU_TOOLCHAIN}/sysroot \
		--gcc-toolchain=${RISCV_GNU_TOOLCHAIN} \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
