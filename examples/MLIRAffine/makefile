#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-runner
LLC := ../../llvm/build/bin/llc
OPT_FLAG := -O0

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so
MLIR_ASYNC_RUNTIME := ../../llvm/build/lib/libmlir_async_runtime.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
MLIR_ASYNC_RUNTIME := ./../llvm/build/lib/libmlir_async_runtime.dylib
MTRIPLE := x86_64-apple-darwin
endif

affine-load-lower:
	@${MLIR_OPT} ./affine-load.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf -convert-cf-to-llvm \
		--finalize-memref-to-llvm --convert-vector-to-llvm --convert-func-to-llvm  -convert-arith-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

affine-load-translate:
	@${MLIR_OPT} ./affine-load.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf -convert-cf-to-llvm \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm -convert-arith-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

affine-load-run:
	@${MLIR_OPT} ./affine-load.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf -convert-cf-to-llvm \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm -convert-arith-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

affine-store-lower:
	@${MLIR_OPT} ./affine-store.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf -convert-cf-to-llvm \
		--finalize-memref-to-llvm --convert-vector-to-llvm --convert-func-to-llvm -convert-arith-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

affine-store-translate:
	@${MLIR_OPT} ./affine-store.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf -convert-cf-to-llvm \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm -convert-arith-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

affine-store-run:
	@${MLIR_OPT} ./affine-store.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf -convert-cf-to-llvm \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm -convert-arith-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

affine-max-lower:
	@${MLIR_OPT} ./affine-max.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf -convert-cf-to-llvm \
		--convert-vector-to-llvm --convert-func-to-llvm -convert-arith-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

affine-max-translate:
	@${MLIR_OPT} ./affine-max.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf -convert-cf-to-llvm \
		--convert-vector-to-llvm --convert-func-to-llvm -convert-arith-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

affine-max-run:
	@${MLIR_OPT} ./affine-max.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf -convert-cf-to-llvm \
		--convert-vector-to-llvm --convert-func-to-llvm -convert-arith-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

affine-parallel-lower:
	@${MLIR_OPT} ./affine-parallel.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf -convert-cf-to-llvm \
		--finalize-memref-to-llvm --convert-vector-to-llvm --convert-func-to-llvm -convert-arith-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

affine-parallel-translate:
	@${MLIR_OPT} ./affine-parallel.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf -convert-cf-to-llvm \
		--finalize-memref-to-llvm --convert-vector-to-llvm --convert-func-to-llvm -convert-arith-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

affine-parallel-run:
	@${MLIR_OPT} ./affine-parallel.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf -convert-cf-to-llvm \
		--finalize-memref-to-llvm --convert-vector-to-llvm --convert-func-to-llvm -convert-arith-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

affine-vector-lower:
	@${MLIR_OPT} ./affine-vector.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf -convert-cf-to-llvm \
		--finalize-memref-to-llvm --convert-vector-to-llvm --convert-func-to-llvm -convert-arith-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

affine-vector-translate:
	@${MLIR_OPT} ./affine-vector.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf -convert-cf-to-llvm \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm -convert-arith-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

affine-vector-run:
	@${MLIR_OPT} ./affine-vector.mlir \
		--convert-vector-to-scf --lower-affine --convert-scf-to-cf -convert-cf-to-llvm \
		--convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm -convert-arith-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
