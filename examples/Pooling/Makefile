#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
OPT_FLAG := -O0

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
endif

pool:
	@${BUDDY_OPT} ./pooling.mlir --pooling-vectorization \
		 --lower-affine --convert-vector-to-scf \
		--convert-scf-to-cf --tensor-bufferize \
		--convert-vector-to-llvm --convert-memref-to-llvm \
		--convert-arith-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

pool-mlir:
	@${MLIR_OPT} ./pooling.mlir -convert-linalg-to-loops \
		--lower-affine --convert-vector-to-scf \
		--convert-scf-to-cf --tensor-bufferize \
		--convert-vector-to-llvm --convert-memref-to-llvm \
		--convert-arith-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

strided:
	@${BUDDY_OPT} ./strided.mlir --pooling-vectorization \
		--lower-affine --convert-vector-to-scf \
		--convert-scf-to-cf --tensor-bufferize \
		--convert-vector-to-llvm --convert-memref-to-llvm \
		--convert-arith-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

strided-mlir:
	@${MLIR_OPT} ./strided.mlir -convert-linalg-to-loops \
		--lower-affine --convert-vector-to-scf \
		--convert-scf-to-cf --tensor-bufferize \
		--convert-vector-to-llvm --convert-memref-to-llvm \
		--convert-arith-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

dilated:
	@${BUDDY_OPT} ./dilated.mlir --pooling-vectorization \
		--lower-affine --convert-vector-to-scf \
		--convert-scf-to-cf --tensor-bufferize \
		--convert-vector-to-llvm --convert-memref-to-llvm \
		--convert-arith-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

dilated-mlir:
	@${MLIR_OPT} ./dilated.mlir -convert-linalg-to-loops \
		--lower-affine --convert-vector-to-scf \
		--convert-scf-to-cf --tensor-bufferize \
		--convert-vector-to-llvm --convert-memref-to-llvm \
		--convert-arith-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

strided-dilated:
	@${BUDDY_OPT} ./strided-dilated.mlir --pooling-vectorization \
		--lower-affine --convert-vector-to-scf \
		--convert-scf-to-cf --tensor-bufferize \
		--convert-vector-to-llvm --convert-memref-to-llvm \
		--convert-arith-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

strided-dilated-mlir:
	@${MLIR_OPT} ./strided-dilated.mlir -convert-linalg-to-loops \
		--lower-affine --convert-vector-to-scf \
		--convert-scf-to-cf --tensor-bufferize \
		--convert-vector-to-llvm --convert-memref-to-llvm \
		--convert-arith-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
