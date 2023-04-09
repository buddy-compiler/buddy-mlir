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
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
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
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
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
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
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
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vector-transpose-lower:
	@${MLIR_OPT} ./vector-transpose.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-transpose-translate:
	@${MLIR_OPT} ./vector-transpose.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-transpose-run:
	@${MLIR_OPT} ./vector-transpose.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vector-shape-cast-lower:
	@${MLIR_OPT} ./vector-shape-cast.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-shape-cast-translate:
	@${MLIR_OPT} ./vector-shape-cast.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-shape-cast-run:
	@${MLIR_OPT} ./vector-shape-cast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		-split-input-file -verify-diagnostics \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vector-type-cast-lower:
	@${MLIR_OPT} ./vector-type-cast.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-type-cast-translate:
	@${MLIR_OPT} ./vector-type-cast.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-type-cast-run:
	@${MLIR_OPT} ./vector-type-cast.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		-split-input-file -verify-diagnostics \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vector-bitcast-lower:
	@${MLIR_OPT} ./vector-bitcast.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-bitcast-translate:
	@${MLIR_OPT} ./vector-bitcast.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-bitcast-run:
	@${MLIR_OPT} ./vector-bitcast.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vector-shuffle-lower:
	@${MLIR_OPT} ./vector-shuffle.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-shuffle-translate:
	@${MLIR_OPT} ./vector-shuffle.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-shuffle-run:
	@${MLIR_OPT} ./vector-shuffle.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		-split-input-file -verify-diagnostics \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vector-splat-lower:
	@${MLIR_OPT} ./vector-splat.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-splat-translate:
	@${MLIR_OPT} ./vector-splat.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-splat-run:
	@${MLIR_OPT} ./vector-splat.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		-split-input-file -verify-diagnostics \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}		

vector-insert-lower:
	@${MLIR_OPT} ./vector-insert.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-insert-translate:
	@${MLIR_OPT} ./vector-insert.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-insert-run:
	@${MLIR_OPT} ./vector-insert.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		-split-input-file -verify-diagnostics \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}		
	
vector-reduction-lower:
	@${MLIR_OPT} ./vector-reduction.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-reduction-translate:
	@${MLIR_OPT} ./vector-reduction.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-reduction-run:
	@${MLIR_OPT} ./vector-reduction.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		-split-input-file -verify-diagnostics \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}	

vector-outerproduct-lower:
	@${MLIR_OPT} ./vector-outerproduct.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-outerproduct-translate:
	@${MLIR_OPT} ./vector-outerproduct.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-outerproduct-run:
	@${MLIR_OPT} ./vector-outerproduct.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		-split-input-file -verify-diagnostics \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}	

vector-create-mask-lower:
	@${MLIR_OPT} ./vector-create-mask.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-create-mask-translate:
	@${MLIR_OPT} ./vector-create-mask.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-create-mask-run:
	@${MLIR_OPT} ./vector-create-mask.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}	

vector-extract-lower:
	@${MLIR_OPT} ./vector-extract.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-extract-translate:
	@${MLIR_OPT} ./vector-extract.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-extract-run:
	@${MLIR_OPT} ./vector-extract.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vector-maskedload-lower:
	@${MLIR_OPT} ./vector-maskedload.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-maskedload-translate:
	@${MLIR_OPT} ./vector-maskedload.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-maskedload-run:
	@${MLIR_OPT} ./vector-maskedload.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		-split-input-file -verify-diagnostics \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vector-maskedstore-lower:
	@${MLIR_OPT} ./vector-maskedstore.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-maskedstore-translate:
	@${MLIR_OPT} ./vector-maskedstore.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-maskedstore-run:
	@${MLIR_OPT} ./vector-maskedstore.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		-split-input-file -verify-diagnostics \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vector-extract-strided-slice-lower:
	@${MLIR_OPT} ./vector-extract-strided-slice.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-extract-strided-slice-translate:
	@${MLIR_OPT} ./vector-extract-strided-slice.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-extract-strided-slice-run:
	@${MLIR_OPT} ./vector-extract-strided-slice.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vector-constant-mask-lower:
	@${MLIR_OPT} ./vector-constant-mask.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-constant-mask-translate:
	@${MLIR_OPT} ./vector-constant-mask.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-constant-mask-run:
	@${MLIR_OPT} ./vector-constant-mask.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
	
vector-expand-load-lower:
	@${MLIR_OPT} ./vector-expand-load.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-expand-load-translate:
	@${MLIR_OPT} ./vector-expand-load.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-expand-load-run:
	@${MLIR_OPT} ./vector-expand-load.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}	

vector-compressstore-lower:
	@${MLIR_OPT} ./vector-compressstore.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-compressstore-translate:
	@${MLIR_OPT} ./vector-compressstore.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-compressstore-run:
	@${MLIR_OPT} ./vector-compressstore.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vector-insert-strided-slice-lower:
	@${MLIR_OPT} ./vector-insert-strided-slice.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-insert-strided-slice-translate:
	@${MLIR_OPT} ./vector-insert-strided-slice.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-insert-strided-slice-run:
	@${MLIR_OPT} ./vector-insert-strided-slice.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}		

vector-scatter-lower:
	@${MLIR_OPT} ./vector-bitcast.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-scatter-translate:
	@${MLIR_OPT} ./vector-bitcast.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-scatter-run:
	@${MLIR_OPT} ./vector-scatter.mlir \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		-split-input-file -verify-diagnostics \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vector-transfer-read-lower:
	@${MLIR_OPT} ./vector-transfer-read.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-transfer-read-translate:
	@${MLIR_OPT} ./vector-transfer-read.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-transfer-read-run:
	@${MLIR_OPT} ./vector-transfer-read.mlir \
	  --convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vector-transfer-write-lower:
	@${MLIR_OPT} ./vector-transfer-write.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-transfer-write-translate:
	@${MLIR_OPT} ./vector-transfer-write.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-transfer-write-run:
	@${MLIR_OPT} ./vector-transfer-write.mlir \
	    --convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vector-contract-lower:
	@${MLIR_OPT} ./vector-contract.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

vector-contract-translate:
	@${MLIR_OPT} ./vector-contract.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

vector-contract-run:
	@${MLIR_OPT} ./vector-contract.mlir \
	    --convert-vector-to-scf --lower-affine --convert-scf-to-cf \
		--convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=i32 \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

test-all: \
	vector-load-run \
	vector-broadcast-run \
	vector-fma-run \
	vector-long-run \
	vector-transpose-run \
	vector-shape-cast-run \
	vector-type-cast-run \
	vector-bitcast-run \
	vector-shuffle-run \
	vector-splat-run \
	vector-insert-run \
	vector-reduction-run \
	vector-outerproduct-run \
	vector-create-mask-run \
	vector-extract-run \
	vector-maskedload-run \
	vector-maskedstore-run \
	vector-extract-strided-slice-run \
	vector-constant-mask-run \
	vector-expand-load-run \
	vector-compressstore-run \
	vector-insert-strided-slice-run \
	vector-scatter-run \
	vector-transfer-read-run \
	vector-transfer-write-run \
	vector-contract-run \
