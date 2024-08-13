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

batch-matmul-1-lower:
	@${MLIR_OPT} ./linalg-batch-matmul-f32.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-arith, tosa-to-linalg, tosa-to-tensor))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-func-bufferize-dynamic-offset \
		-arith-bufferize \
		-func-bufferize \
		-tensor-bufferize \
		-linalg-bufferize \
		-finalizing-bufferize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-llvm-request-c-wrappers \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-expand-strided-metadata \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o ./log.mlir


batch-matmul-1-translate:
	@${MLIR_OPT} ./linalg-batch-matmul-f32.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-arith, tosa-to-linalg, tosa-to-tensor))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-func-bufferize-dynamic-offset \
		-arith-bufferize \
		-func-bufferize \
		-tensor-bufferize \
		-linalg-bufferize \
		-finalizing-bufferize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-llvm-request-c-wrappers \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-expand-strided-metadata \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

batch-matmul-1-run:
	@${MLIR_OPT} ./linalg-batch-matmul-f32.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-arith, tosa-to-linalg, tosa-to-tensor))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-func-bufferize-dynamic-offset \
		-arith-bufferize \
		-func-bufferize \
		-tensor-bufferize \
		-linalg-bufferize \
		-finalizing-bufferize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-llvm-request-c-wrappers \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-expand-strided-metadata \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}


batch-matmul-2-lower:
	@${MLIR_OPT} ./linalg-batch-matmul-f32-2.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-arith, tosa-to-linalg, tosa-to-tensor))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-func-bufferize-dynamic-offset \
		-arith-bufferize \
		-func-bufferize \
		-tensor-bufferize \
		-linalg-bufferize \
		-finalizing-bufferize \
		-batchmatmul-optimize \
		-o ./log.mlir


batch-matmul-2-translate:
	@${MLIR_OPT} ./linalg-batch-matmul-f32-2.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-arith, tosa-to-linalg, tosa-to-tensor))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-func-bufferize-dynamic-offset \
		-arith-bufferize \
		-func-bufferize \
		-tensor-bufferize \
		-linalg-bufferize \
		-finalizing-bufferize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-llvm-request-c-wrappers \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-expand-strided-metadata \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

batch-matmul-2-run:
	@${MLIR_OPT} ./linalg-batch-matmul-f32-2.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-arith, tosa-to-linalg, tosa-to-tensor))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-func-bufferize-dynamic-offset \
		-arith-bufferize \
		-func-bufferize \
		-tensor-bufferize \
		-linalg-bufferize \
		-finalizing-bufferize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-llvm-request-c-wrappers \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-expand-strided-metadata \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}