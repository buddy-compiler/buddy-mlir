#!/bin/bash
BUDDY_OPT := ../../build-timing/bin/buddy-opt
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

timing-llama-lower:
	@${BUDDY_OPT} ./timing-llama.mlir \
		--timing --lower-timing | \
	${MLIR_OPT} -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} -arith-expand -eliminate-empty-tensors \
    -empty-tensor-to-alloc-tensor -one-shot-bufferize \
    -matmul-paralell-vectorization-optimize -batchmatmul-optimize \
    -convert-linalg-to-affine-loops -affine-loop-fusion \
    -affine-parallelize -lower-affine \
    -convert-scf-to-openmp -func-bufferize-dynamic-offset \
    -tensor-bufferize -arith-bufferize \
    -buffer-deallocation -finalizing-bufferize \
    -convert-vector-to-scf -expand-strided-metadata \
    -cse -convert-vector-to-llvm \
    -memref-expand -arith-expand \
    -convert-arith-to-llvm -finalize-memref-to-llvm \
    -convert-scf-to-cf -llvm-request-c-wrappers \
    -convert-openmp-to-llvm -convert-arith-to-llvm \
    -convert-math-to-llvm -convert-math-to-libm \
    -convert-func-to-llvm -reconcile-unrealized-casts \
	-o ./log.mlir

timing-llama-translate:
	@${BUDDY_OPT} ./timing-llama.mlir \
		--timing --lower-timing | \
	${MLIR_OPT} -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} -arith-expand -eliminate-empty-tensors \
    -empty-tensor-to-alloc-tensor -one-shot-bufferize \
    -matmul-paralell-vectorization-optimize -batchmatmul-optimize \
    -convert-linalg-to-affine-loops -affine-loop-fusion \
    -affine-parallelize -lower-affine \
    -convert-scf-to-openmp -func-bufferize-dynamic-offset \
    -tensor-bufferize -arith-bufferize \
    -buffer-deallocation -finalizing-bufferize \
    -convert-vector-to-scf -expand-strided-metadata \
    -cse -convert-vector-to-llvm \
    -memref-expand -arith-expand \
    -convert-arith-to-llvm -finalize-memref-to-llvm \
    -convert-scf-to-cf -llvm-request-c-wrappers \
    -convert-openmp-to-llvm -convert-arith-to-llvm \
    -convert-math-to-llvm -convert-math-to-libm \
    -convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

timing-llama-run:
	@${BUDDY_OPT} ./timing-llama.mlir \
		--timing --lower-timing | \
	${MLIR_OPT} -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} -arith-expand -eliminate-empty-tensors \
    -empty-tensor-to-alloc-tensor -one-shot-bufferize \
    -matmul-paralell-vectorization-optimize -batchmatmul-optimize \
    -convert-linalg-to-affine-loops -affine-loop-fusion \
    -affine-parallelize -lower-affine \
    -convert-scf-to-openmp -func-bufferize-dynamic-offset \
    -tensor-bufferize -arith-bufferize \
    -buffer-deallocation -finalizing-bufferize \
    -convert-vector-to-scf -expand-strided-metadata \
    -cse -convert-vector-to-llvm \
    -memref-expand -arith-expand \
    -convert-arith-to-llvm -finalize-memref-to-llvm \
    -convert-scf-to-cf -llvm-request-c-wrappers \
    -convert-openmp-to-llvm -convert-arith-to-llvm \
    -convert-math-to-llvm -convert-math-to-libm \
    -convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

timing-conv2d-lower:
	@${BUDDY_OPT} ./timing-conv2d.mlir \
		--timing --lower-timing | \
    ${MLIR_OPT} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts \
		-o ./log.mlir

timing-conv2d-translate:
	@${BUDDY_OPT} ./timing-conv2d.mlir \
		--timing --lower-timing | \
	${MLIR_OPT} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

timing-conv2d-run:
	@${BUDDY_OPT} ./timing-conv2d.mlir \
		--timing --lower-timing | \
	${MLIR_OPT} ${MLIR_OPT_OPTIONS} \
		-convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}



timing-llama2-lower:
	@${BUDDY_OPT} ./timing-llama.mlir --timing -o log.mlir