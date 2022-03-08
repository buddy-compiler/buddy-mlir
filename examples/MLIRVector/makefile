#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
LLC := ../../llvm/build/bin/llc
OPT_FLAG := -O0

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
MTRIPLE := x86_64-apple-darwin
endif

vector-load-lower:
	@${MLIR_OPT} ./vector-load.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-load-translate:
	@${MLIR_OPT} ./vector-load.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-load-run:
	@${MLIR_OPT} ./vector-load.mlir \
	    --convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vector-broadcast-lower:
	@${MLIR_OPT} ./vector-broadcast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-broadcast-translate:
	@${MLIR_OPT} ./vector-broadcast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-broadcast-asm-x86:
	@${MLIR_OPT} ./vector-broadcast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f\
		--filetype=asm -o log.s

vector-broadcast-asm-rv:
	@${MLIR_OPT} ./vector-broadcast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

vector-broadcast-run:
	@${MLIR_OPT} ./vector-broadcast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vector-fma-lower:
	@${MLIR_OPT} ./vector-fma.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-fma-translate:
	@${MLIR_OPT} ./vector-fma.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-fma-asm-x86:
	@${MLIR_OPT} ./vector-fma.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f\
		--filetype=asm -o log.s

vector-fma-asm-rv:
	@${MLIR_OPT} ./vector-fma.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

vector-fma-run:
	@${MLIR_OPT} ./vector-fma.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vector-long-lower:
	@${MLIR_OPT} ./vector-long.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-long-translate:
	@${MLIR_OPT} ./vector-long.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-long-asm-x86:
	@${MLIR_OPT} ./vector-long.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f\
		--filetype=asm -o log.s

vector-long-asm-rv:
	@${MLIR_OPT} ./vector-long.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

vector-long-run:
	@${MLIR_OPT} ./vector-long.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
