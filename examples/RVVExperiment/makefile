#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
LLC := ../../llvm/build/bin/llc
OPT := ../../llvm/build/bin/opt
OPT_FLAG := -O3

RISCV_GNU_TOOLCHAIN := ../../thirdparty/build-riscv-gnu-toolchain
RISCV_GNU_TOOLCHAIN_SYSROOT := ../../thirdparty/build-riscv-gnu-toolchain/sysroot
QEMU := ../../thirdparty/qemu/build/riscv64-linux-user/qemu-riscv64
LOCAL_CLANG := ../../thirdparty/build-local-clang/bin/clang
CROSS_LLI := ../../thirdparty/build-cross-clang/bin/lli
CROSS_MLIR_C_RUNNER_UTILS := ../../thirdparty/build-cross-mlir/lib/libmlir_c_runner_utils.so
CROSS_MLIR_LIB := ../../thirdparty/build-cross-mlir/lib

MLIR_VECTOR_EXAMPLES := ../MLIRVector

.SECONDEXPANSION:
all-run: $$(run-targets)

rvv-broadcast-lower:
	@${MLIR_OPT} ./rvv-broadcast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-broadcast-translate:
	@${MLIR_OPT} ./rvv-broadcast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-broadcast-asm-rv:
	@${MLIR_OPT} ./rvv-broadcast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

run-targets += rvv-fma-run
rvv-broadcast-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-broadcast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-fma-lower:
	@${MLIR_OPT} ./rvv-fma.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-fma-translate:
	@${MLIR_OPT} ./rvv-fma.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-fma-asm-x86:
	@${MLIR_OPT} ./rvv-fma.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f\
		--filetype=asm -o log.s

rvv-fma-asm-rv:
	@${MLIR_OPT} ./rvv-fma.mlir \
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

rvv-long-lower:
	@${MLIR_OPT} ./rvv-long.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-long-translate:
	@${MLIR_OPT} ./rvv-long.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-long-asm-x86:
	@${MLIR_OPT} ./rvv-long.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f\
		--filetype=asm -o log.s

rvv-long-asm-rv:
	@${MLIR_OPT} ./rvv-long.mlir \
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

rvv-scalable-lower:
	@${MLIR_OPT} ./rvv-scalable.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-scalable-translate:
	@${MLIR_OPT} ./rvv-scalable.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-scalable-asm-rv:
	@${MLIR_OPT} ./rvv-scalable.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

rvv-scalable-run-128:
	@${MLIR_OPT} ./rvv-scalable.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-scalable-aot-128:
	@${MLIR_OPT} ./rvv-scalable.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=128 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 a.out

rvv-scalable-run-256:
	@${MLIR_OPT} ./rvv-scalable.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=256 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-scalable-aot-256:
	@${MLIR_OPT} ./rvv-scalable.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} -mtriple riscv64 -target-abi lp64d -mattr=+m,+d,+v -riscv-v-vector-bits-min=256 --filetype=obj -o log.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc log.o -mabi=lp64d \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=256 a.out

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

run-targets += rvv-c-setvl-run
rvv-c-setvl-run:
	@${LOCAL_CLANG} -march=rv64gcv --target=riscv64-unknown-linux-gnu \
		--sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT} --gcc-toolchain=${RISCV_GNU_TOOLCHAIN} \
		./rvv-c-setvl.c -fPIC -S -emit-llvm -o log.ll
	@${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true \
		${CROSS_LLI} --entry-function=main --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic log.ll

run-targets += rvv-transpose-run
rvv-transpose-run:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-transpose.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}
