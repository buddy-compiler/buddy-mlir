#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
LLC := ../../llvm/build/bin/llc
OPT := ../../llvm/build/bin/opt
OPT_FLAG := -O3

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
MTRIPLE := x86_64-apple-darwin
endif

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

rvv-broadcast-run:
	@${MLIR_OPT} ./rvv-broadcast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

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

rvv-fma-run:
	@${MLIR_OPT} ./rvv-fma.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

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

rvv-long-run:
	@${MLIR_OPT} ./rvv-long.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

rvv-scalable-lower:
	@${MLIR_OPT} ./rvv-scalable.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

rvv-scalable-asm-rv:
	@${MLIR_OPT} ./rvv-scalable.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

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
