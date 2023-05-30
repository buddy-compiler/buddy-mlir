#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
LLC := ../../llvm/build/bin/llc
OPT_FLAG := -O0

RISCV_GNU_TOOLCHAIN := ../../thirdparty/build-riscv-gnu-toolchain
RISCV_GNU_TOOLCHAIN_SYSROOT := ../../thirdparty/build-riscv-gnu-toolchain/sysroot
QEMU := ../../thirdparty/qemu/build/riscv64-linux-user/qemu-riscv64
LOCAL_CLANG := ../../thirdparty/build-local-clang/bin/clang
CROSS_LLI := ../../thirdparty/build-cross-clang/bin/lli
CROSS_MLIR_CPU_RUNNER := ../../thirdparty/build-cross-mlir/bin/mlir-cpu-runner
CROSS_MLIR_C_RUNNER_UTILS := ../../thirdparty/build-cross-mlir/lib/libmlir_c_runner_utils.so
CROSS_MLIR_RUNNER_UTILS := ../../thirdparty/build-cross-mlir/lib/libmlir_runner_utils.so
CROSS_MLIR_LIB := ../../thirdparty/build-cross-mlir/lib

rvv-setvl-lower:
	@${BUDDY_OPT} ./rvv-setvl.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-rvv \
		-convert-vector-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts \
		-o ./log.mlir

rvv-setvl-translate:
	@${BUDDY_OPT} ./rvv-setvl.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-rvv \
		-convert-vector-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir -o log.ll

rvv-setvl-128-run:
	@${BUDDY_OPT} ./rvv-setvl.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-rvv \
		-convert-vector-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} -march=riscv64 -mattr=+m,+d,+v \
		-dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-setvl-256-run:
	@${BUDDY_OPT} ./rvv-setvl.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-rvv \
		-convert-vector-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=256 \
	${CROSS_LLI} -march=riscv64 -mattr=+m,+d,+v \
		-dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-rsqrt-lower:
	@${BUDDY_OPT} ./rvv-rsqrt.mlir \
		--convert-scf-to-cf \
		--convert-math-to-llvm \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts \
		-o ./log.mlir

rvv-rsqrt-translate:
	@${BUDDY_OPT} ./rvv-rsqrt.mlir \
		--convert-scf-to-cf \
		--convert-math-to-llvm \
		--lower-rvv --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
		${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll

rvv-rsqrt-aot:
	@${BUDDY_OPT} ./rvv-rsqrt.mlir \
		-convert-scf-to-cf \
		--convert-math-to-llvm \
		--lower-rvv -convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm\
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-mul-add-lower:
	@${BUDDY_OPT} ./rvv-mul-add.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
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
		-lower-rvv \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} \
			-cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} -march=riscv64 -mattr=+m,+d,+v \
		-dlopen=${CROSS_MLIR_C_RUNNER_UTILS} \
		-dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-stripmining-lower:
	@${BUDDY_OPT} ./rvv-stripmining.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-rvv \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts \
		-o ./log.mlir

rvv-stripmining-translate:
	@${BUDDY_OPT} ./rvv-stripmining.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-rvv \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir -o log.ll

rvv-stripmining-run:
	@${BUDDY_OPT} ./rvv-stripmining.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-rvv \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} \
			-cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} -march=riscv64 -mattr=+m,+d,+v \
		-dlopen=${CROSS_MLIR_C_RUNNER_UTILS} \
		-dlopen=${CROSS_MLIR_RUNNER_UTILS}

rvv-stripmining-aot:
	@${BUDDY_OPT} ./rvv-stripmining.mlir \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-rvv \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o \
		-mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out
