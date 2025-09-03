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
LIB_OMP := ../../llvm/build/lib/libomp.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
MTRIPLE := x86_64-apple-darwin
endif

next-attention-lower:
	@${MLIR_OPT} ./next-attention.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts \
		-o ./log.mlir

next-attention-translate:
	@${MLIR_OPT} ./next-attention.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

next-attention-run:
	@${MLIR_OPT} ./next-attention.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		--one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-attention-loop-run:
	@${MLIR_OPT} ./next-attention-loop.mlir \
		-affine-loop-fusion \
		-lower-affine \
		--one-shot-bufferize="bufferize-function-boundaries" \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-attention-fusion-run:
	@${MLIR_OPT} ./next-attention-fusion.mlir \
		-affine-loop-fusion \
		-lower-affine \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-sigmoid-run:
	@${MLIR_OPT} ./next-sigmoid.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-eliminate-add-zero-run:
	@${MLIR_OPT} ./next-eliminate-add-zero.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-rope-run:
	@${MLIR_OPT} ./next-rope.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-lower-affine \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}


next-eliminate-identity-run:
	@${MLIR_OPT} ./next-eliminate-identity.mlir \
        -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
    ${MLIR_OPT} \
        -arith-expand \
        -eliminate-empty-tensors \
        -empty-tensor-to-alloc-tensor \
        -one-shot-bufferize="bufferize-function-boundaries" \
        -convert-linalg-to-affine-loops \
        -affine-loop-fusion \
        -lower-affine \
        -convert-vector-to-scf \
        -expand-strided-metadata \
        -convert-vector-to-llvm \
        -memref-expand \
        -arith-expand \
        -convert-arith-to-llvm \
        -finalize-memref-to-llvm \
        -convert-scf-to-cf \
		-convert-cf-to-llvm \
        -convert-openmp-to-llvm \
        -convert-arith-to-llvm \
        -convert-math-to-llvm \
        -convert-math-to-libm  \
        -convert-func-to-llvm \
        -reconcile-unrealized-casts | \
    ${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
        -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-sgemm-run:
	@${MLIR_OPT} ./next-sgemm.mlir \
		-convert-linalg-to-loops \
		-cse \
        -lower-affine \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
    ${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
        -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-transpose-lower:
	@${MLIR_OPT} ./next-transpose.mlir \
        -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
    ${MLIR_OPT} \
        -arith-expand \
        -eliminate-empty-tensors \
        -empty-tensor-to-alloc-tensor \
        -one-shot-bufferize="bufferize-function-boundaries" \
        -o log.mlir

next-transpose-run:
	@${MLIR_OPT} ./next-transpose.mlir \
        -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
    ${MLIR_OPT} \
        -arith-expand \
        -eliminate-empty-tensors \
        -empty-tensor-to-alloc-tensor \
        -one-shot-bufferize="bufferize-function-boundaries" \
        -convert-linalg-to-affine-loops \
        -affine-loop-fusion \
        -lower-affine \
        -convert-vector-to-scf \
        -expand-strided-metadata \
        -convert-vector-to-llvm \
        -memref-expand \
        -arith-expand \
        -convert-arith-to-llvm \
        -finalize-memref-to-llvm \
        -convert-scf-to-cf \
		-convert-cf-to-llvm \
        -convert-openmp-to-llvm \
        -convert-arith-to-llvm \
        -convert-math-to-llvm \
        -convert-math-to-libm  \
        -convert-func-to-llvm \
        -reconcile-unrealized-casts | \
    ${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
        -shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS}

next-transpose-vec-manual-run:
	@${MLIR_OPT} ./next-transpose-vec-manual.mlir \
        -convert-linalg-to-affine-loops \
        -affine-loop-fusion \
        -lower-affine \
		-convert-scf-to-openmp \
        -convert-vector-to-scf \
        -expand-strided-metadata \
        -convert-vector-to-llvm \
        -memref-expand \
        -arith-expand \
        -convert-arith-to-llvm \
        -finalize-memref-to-llvm \
        -convert-scf-to-cf \
		-convert-cf-to-llvm \
        -convert-openmp-to-llvm \
        -convert-arith-to-llvm \
        -convert-math-to-llvm \
        -convert-math-to-libm  \
        -convert-func-to-llvm \
        -reconcile-unrealized-casts | \
    ${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
        -shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS}

next-embedding-lower:
	@${MLIR_OPT} ./next-embedding.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-o log.mlir

next-embedding-run:
	@${MLIR_OPT} ./next-embedding.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-lower-affine \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-mask-run:
	@${MLIR_OPT} ./next-mask.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-norm-run:
	@${MLIR_OPT} ./next-norm.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-lower-affine \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-mhsa-qkv-run:
	@${MLIR_OPT} ./next-mhsa-qkv.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-lower-affine \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-mhsa-qkv-parallel-vec-run:
	@${MLIR_OPT} ./next-mhsa-qkv.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-matmul-parallel-vectorization-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-lower-affine \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-mhsa-qkv-fusion-lower:
	@${MLIR_OPT} ./next-mhsa-qkv-fusion.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map" \
		-matmul-transpose-b-vectorization \
		-o log.mlir

next-mhsa-qkv-fusion-run:
	@${MLIR_OPT} ./next-mhsa-qkv-fusion.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-lower-affine \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-mhsa-qkv-fusion-vec-run:
	@${MLIR_OPT} ./next-mhsa-qkv-fusion.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map" \
		-matmul-transpose-b-vectorization \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-mhsa-core-run:
	@${MLIR_OPT} ./next-mhsa-core.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-lower-affine \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-mhsa-context-run:
	@${MLIR_OPT} ./next-mhsa-context.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-lower-affine \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-ffn-run:
	@${MLIR_OPT} ./next-ffn.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-lower-affine \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-ffn-parallel-vec-run:
	@${MLIR_OPT} ./next-ffn.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-matmul-parallel-vectorization-optimize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-lower-affine \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

pooling-nhwc-max-vec-run:
	@${BUDDY_OPT} ./pooling-nhwc-max-vec.mlir \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-compass-lower:
	@${MLIR_OPT} ./next-compass.mlir \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts  \
		-o log.mlir

next-compass-run:
	@${MLIR_OPT} ./next-compass.mlir \
    -arith-expand \
    -eliminate-empty-tensors \
    -empty-tensor-to-alloc-tensor \
    -one-shot-bufferize="bufferize-function-boundaries" \
    -convert-linalg-to-affine-loops \
    -affine-loop-fusion \
    -lower-affine \
    -convert-vector-to-scf \
    -expand-strided-metadata \
    -convert-vector-to-llvm \
    -memref-expand \
    -arith-expand \
    -convert-arith-to-llvm \
    -finalize-memref-to-llvm \
    -convert-scf-to-cf \
    -convert-openmp-to-llvm \
    -convert-arith-to-llvm \
    -convert-math-to-llvm \
    -convert-math-to-libm  \
    -convert-func-to-llvm \
    -reconcile-unrealized-casts  | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS}

tosa-matmul-transpose2-lower:
	@${BUDDY_OPT} ./tosa-matmultranspose2.mlir \
			-transpose-fusion-vectorization \
			-o log.mlir

tosa-matmul-transpose2-run:
	@${BUDDY_OPT} ./tosa-matmultranspose2.mlir \
			-pass-pipeline "builtin.module(transpose-fusion-vectorization, func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" | \
    ${BUDDY_OPT} \
		-eliminate-empty-tensors \
		-convert-tensor-to-linalg \
		-linalg-bufferize \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

tosa-matmul-transpose2-vec-run:
	@${BUDDY_OPT} ./tosa-matmultranspose2-vec.mlir\
		-convert-linalg-to-affine-loops \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm  \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
