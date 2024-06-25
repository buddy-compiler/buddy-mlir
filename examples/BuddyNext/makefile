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

next-attention-lower:
	@${MLIR_OPT} ./next-attention.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
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
		-reconcile-unrealized-casts \
		-o ./log.mlir

next-attention-translate:
	@${MLIR_OPT} ./next-attention.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
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
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

next-attention-run:
	@${MLIR_OPT} ./next-attention.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
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
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-attention-loop-run:
	@${MLIR_OPT} ./next-attention-loop.mlir \
		-affine-loop-fusion \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
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
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-attention-fusion-run:
	@${MLIR_OPT} ./next-attention-fusion.mlir \
		-affine-loop-fusion \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
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
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
