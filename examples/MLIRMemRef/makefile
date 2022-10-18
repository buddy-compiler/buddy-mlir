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
MLIR_ASYNC_RUNTIME := ../../llvm/build/lib/libmlir_async_runtime.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
MLIR_ASYNC_RUNTIME := ./../llvm/build/lib/libmlir_async_runtime.dylib
MTRIPLE := x86_64-apple-darwin
endif

memref-memory-lower:
	@${MLIR_OPT} ./memref-memory.mlir \
		-convert-vector-to-llvm \
		-convert-memref-to-llvm -convert-arith-to-llvm  \
		-convert-func-to-llvm -reconcile-unrealized-casts \
		-o ./log.mlir

memref-memory-translate:
	@${MLIR_OPT} ./memref-memory.mlir \
		-convert-vector-to-llvm \
		-convert-memref-to-llvm -convert-arith-to-llvm  \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

memref-memory-run:
	@${MLIR_OPT} ./memref-memory.mlir --lower-affine  \
		-convert-vector-to-llvm \
		-convert-memref-to-llvm -convert-arith-to-llvm  \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

memref-subview-lower:
	@${MLIR_OPT} ./memref-subview.mlir \
		-convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts \
		-o ./log.mlir

memref-subview-translate:
	@${MLIR_OPT} ./memref-subview.mlir \
		-convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

memref-subview-run:
	@${MLIR_OPT} ./memref-subview.mlir --lower-affine  \
		-convert-memref-to-llvm -convert-arith-to-llvm  \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

memref-dim-lower:
	@${MLIR_OPT} ./memref-dim.mlir \
		-convert-memref-to-llvm -convert-arith-to-llvm -convert-vector-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts \
		-o ./log.mlir

memref-dim-translate:
	@${MLIR_OPT} ./memref-dim.mlir \
		-convert-memref-to-llvm -convert-arith-to-llvm -convert-vector-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

memref-dim-run:
	@${MLIR_OPT} ./memref-dim.mlir --lower-affine  \
		-convert-memref-to-llvm -convert-arith-to-llvm  -convert-vector-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

memref-rank-lower:
	@${MLIR_OPT} ./memref-rank.mlir \
		-convert-memref-to-llvm -convert-arith-to-llvm -convert-vector-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts \
		-o ./log.mlir

memref-rank-translate:
	@${MLIR_OPT} ./memref-rank.mlir \
		-convert-memref-to-llvm -convert-arith-to-llvm -convert-vector-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

memref-rank-run:
	@${MLIR_OPT} ./memref-rank.mlir --lower-affine  \
		-convert-memref-to-llvm -convert-arith-to-llvm -convert-vector-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

memref-expand-shape-lower:
	@${MLIR_OPT} ./memref-expand-shape.mlir \
		-convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts \
		-o ./log.mlir

memref-expand-shape-translate:
	@${MLIR_OPT} ./memref-expand-shape.mlir \
		-convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

memref-expand-shape-run:
	@${MLIR_OPT} ./memref-expand-shape.mlir --lower-affine  \
		-convert-memref-to-llvm -convert-arith-to-llvm  \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

memref-reinterpret-cast-lower:
	@${MLIR_OPT} ./memref-reinterpret-cast.mlir \
		-convert-memref-to-llvm -convert-arith-to-llvm -convert-scf-to-cf \
		-convert-func-to-llvm -reconcile-unrealized-casts \
		-o ./log.mlir

memref-reinterpret-cast-translate:
	@${MLIR_OPT} ./memref-reinterpret-cast.mlir \
		-convert-memref-to-llvm -convert-arith-to-llvm -convert-scf-to-cf \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

memref-reinterpret-cast-run:
	@${MLIR_OPT} ./memref-reinterpret-cast.mlir --lower-affine  \
		-convert-memref-to-llvm -convert-arith-to-llvm -convert-scf-to-cf \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
