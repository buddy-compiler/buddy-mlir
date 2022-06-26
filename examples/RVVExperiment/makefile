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
# 
# Vector ops
#

# FIXME: error: failed to legalize operation 'builtin.unrealized_conversion_cast' that was explicitly marked illegal
# rvv-load-lower:
# 	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-load.mlir \
# 		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
# 		--reconcile-unrealized-casts -o ./log.mlir

# rvv-load-translate:
# 	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-load.mlir \
# 		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
# 		--reconcile-unrealized-casts | \
# 	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

# rvv-load-asm-rv:
# 	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-load.mlir \
# 		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
# 		--reconcile-unrealized-casts | \
# 	${MLIR_TRANSLATE} --mlir-to-llvmir | \
# 	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
# 		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
# 		--filetype=asm -o log.s

# run-targets += rvv-load-run
# rvv-load-run:
# 	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-load.mlir \
# 		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
# 		--reconcile-unrealized-casts | \
# 	${MLIR_TRANSLATE} --mlir-to-llvmir | \
# 	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
# 	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
# 		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-broadcast-lower:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-broadcast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-broadcast-translate:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-broadcast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-broadcast-asm-rv:
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

rvv-fma-lower:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/rvv-fma.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-fma-translate:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/rvv-fma.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-fma-asm-rv:
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

rvv-long-lower:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/rvv-long.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-long-translate:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/rvv-long.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-long-asm-rv:
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
rvv-transpose-lower:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-transpose.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-transpose-translate:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-transpose.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-transpose-asm-rv:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-transpose.mlir \
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

rvv-shape-cast-lower:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-shape-cast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-shape-cast-translate:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-shape-cast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-shape-cast-asm-rv:
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

rvv-bitcast-lower:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-bitcast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-bitcast-translate:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-bitcast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-bitcast-asm-rv:
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

rvv-shuffle-lower:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-shuffle.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-shuffle-translate:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-shuffle.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-shuffle-asm-rv:
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

rvv-splat-lower:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-splat.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-splat-translate:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-splat.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-splat-asm-rv:
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

rvv-insert-lower:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-insert.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-insert-translate:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-insert.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-insert-asm-rv:
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

rvv-reduction-lower:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-reduction.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-reduction-translate:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-reduction.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-reduction-asm-rv:
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

rvv-outerproduct-lower:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-outerproduct.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-outerproduct-translate:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-outerproduct.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-outerproduct-asm-rv:
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

rvv-createmask-lower:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-createmask.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-createmask-translate:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-createmask.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-createmask-asm-rv:
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

rvv-extract-lower:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-extract.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-extract-translate:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-extract.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-extract-asm-rv:
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

rvv-maskedload-lower:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-maskedload.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-maskedload-translate:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-maskedload.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-maskedload-asm-rv:
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

# FIXME: JIT session error: Symbols not found: [ printMemrefI32 ]
# rvv-maskedstore-lower:
# 	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-maskedstore.mlir \
# 		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
# 		--reconcile-unrealized-casts -o ./log.mlir

# rvv-maskedstore-translate:
# 	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-maskedstore.mlir \
# 		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
# 		--reconcile-unrealized-casts | \
# 	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

# rvv-maskedstore-asm-rv:
# 	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-maskedstore.mlir \
# 		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
# 		--reconcile-unrealized-casts | \
# 	${MLIR_TRANSLATE} --mlir-to-llvmir | \
# 	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
# 		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
# 		--filetype=asm -o log.s

# run-targets += rvv-maskedstore-run
# rvv-maskedstore-run:
# 	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-maskedstore.mlir \
# 		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
# 		--reconcile-unrealized-casts | \
# 	${MLIR_TRANSLATE} --mlir-to-llvmir | \
# 	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
# 	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
# 		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-extract-strided-slice-lower:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-extract-strided-slice.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-extract-strided-slice-translate:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-extract-strided-slice.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-extract-strided-slice-asm-rv:
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

rvv-constant-mask-lower:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-constant-mask.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-constant-mask-translate:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-constant-mask.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-constant-mask-asm-rv:
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

rvv-expand-load-lower:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-expand-load.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-expand-load-translate:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-expand-load.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-expand-load-asm-rv:
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
# FIXME: JIT session error: Symbols not found: [ printMemrefI32 ]
# rvv-compressstore-lower:
# 	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-compressstore.mlir \
# 		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
# 		--reconcile-unrealized-casts -o ./log.mlir

# rvv-compressstore-translate:
# 	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-compressstore.mlir \
# 		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
# 		--reconcile-unrealized-casts | \
# 	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

# rvv-compressstore-asm-rv:
# 	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-compressstore.mlir \
# 		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
# 		--reconcile-unrealized-casts | \
# 	${MLIR_TRANSLATE} --mlir-to-llvmir | \
# 	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
# 		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
# 		--filetype=asm -o log.s

# run-targets += rvv-compressstore-run
# rvv-compressstore-run:
# 	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-compressstore.mlir \
# 		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
# 		--reconcile-unrealized-casts | \
# 	${MLIR_TRANSLATE} --mlir-to-llvmir | \
# 	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
# 	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
# 		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}

rvv-insert-strided-slice-lower:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-insert-strided-slice.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-insert-strided-slice-translate:
	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-insert-strided-slice.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

rvv-insert-strided-slice-asm-rv:
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

# FIXME: JIT session error: Symbols not found: [ printMemrefI32 ]
# rvv-scatter-lower:
# 	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-scatter.mlir \
# 		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
# 		--reconcile-unrealized-casts -o ./log.mlir

# rvv-scatter-translate:
# 	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-scatter.mlir \
# 		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
# 		--reconcile-unrealized-casts | \
# 	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

# rvv-scatter-asm-rv:
# 	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-scatter.mlir \
# 		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
# 		--reconcile-unrealized-casts | \
# 	${MLIR_TRANSLATE} --mlir-to-llvmir | \
# 	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
# 		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
# 		--filetype=asm -o log.s

# run-targets += rvv-scatter-run
# rvv-scatter-run:
# 	@${MLIR_OPT} ${MLIR_VECTOR_EXAMPLES}/vector-scatter.mlir \
# 		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
# 		--reconcile-unrealized-casts | \
# 	${MLIR_TRANSLATE} --mlir-to-llvmir | \
# 	${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 \
# 	${CROSS_LLI} --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
# 		--dlopen=${CROSS_MLIR_C_RUNNER_UTILS}
