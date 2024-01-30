#!/bin/bash
MLIR_OPT := ../../llvm/build/bin/mlir-opt
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_TRANSLATE := ../../llvm-project/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm-project/build/bin/mlir-cpu-runner
LLC := ../../llvm-project/build/bin/llc
LLVM_AS := ../../llvm-project/build/bin/llvm-as
OPT_FLAG := -O0

MLIR_RUNNER_UTILS := ../../llvm-project/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm-project/build/lib/libmlir_c_runner_utils.so
MLIR_ASYNC_RUNTIME := ../../llvm-project/build/lib/libmlir_async_runtime.so
MLIR_CUDA_RUNTIME := ../../llvm-project/build/lib/libmlir_cuda_runtime.so
MTRIPLE := x86_64-unknown-linux-gnu

forward-lower:
	@${MLIR_OPT} ./forward.mlir \
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
          ${BUDDY_OPT} \
            -arith-expand \
            -eliminate-empty-tensors \
            -empty-tensor-to-alloc-tensor \
            -linalg-bufferize \
            -matmul-paralell-vectorization-optimize \
            -batchmatmul-optimize \
            -convert-linalg-to-affine-loops \
            -affine-loop-fusion \
            -affine-parallelize \
            -lower-affine \
            -convert-scf-to-openmp \
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
            -llvm-request-c-wrappers \
            -convert-openmp-to-llvm \
            -convert-arith-to-llvm \
            -convert-math-to-llvm \
            -convert-math-to-libm  \
            -convert-func-to-llvm \
            -reconcile-unrealized-casts | \
        ${MLIR_TRANSLATE} -mlir-to-llvmir | \
        ${LLVM_AS} | \
        ${LLC} -filetype=obj -relocation-model=pic -O3 -o ./forward.o

subgraph-lower:
	@${MLIR_OPT} ./subgraph0.mlir \
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
          ${BUDDY_OPT} -debug-only=wlq \
            -arith-expand \
            -eliminate-empty-tensors \
            -empty-tensor-to-alloc-tensor \
            -one-shot-bufferize \
            -convert-linalg-to-affine-loops \
            -affine-loop-fusion \
            -affine-parallelize \
            -lower-affine \
            -convert-scf-to-openmp \
            -func-bufferize-dynamic-offset \
            -tensor-bufferize \
            -arith-bufferize \
            -buffer-deallocation \
            -finalizing-bufferize \
            -convert-vector-to-scf \
            -expand-strided-metadata \
            -cse \
            -convert-vector-to-llvm \
            -memref-expand \
            -mlir-print-ir-after-all