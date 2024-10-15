#!/bin/bash
BUDDY_BUILD_DIR := ../../build/
LLVM_BUILD_DIR := ../../llvm/build/
BUDDY_OPT := ${BUDDY_BUILD_DIR}/bin/buddy-opt
MLIR_OPT := ${LLVM_BUILD_DIR}/bin/mlir-opt
MLIR_TRANSLATE := ${LLVM_BUILD_DIR}/bin/mlir-translate
MLIR_CPU_RUNNER := ${LLVM_BUILD_DIR}/bin/mlir-cpu-runner
LLC := ${LLVM_BUILD_DIR}/bin/llc
OPT_FLAG := -O3

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.so
MLIR_ASYNC_RUNTIME := ${LLVM_BUILD_DIR}/lib/libmlir_async_runtime.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.dylib
MLIR_ASYNC_RUNTIME := ${LLVM_BUILD_DIR}/lib/libmlir_async_runtime.dylib
MTRIPLE := x86_64-apple-darwin
endif

buddy-gpu-matmul-lower:
	@${BUDDY_OPT} subgraph0.mlir \
		-transform-preload-library="transform-library-paths=transform.mlir" \
		-transform-interpreter="entry-point=codegen" \
		-o log.mlir

buddy-gpu-matmul:
	@${BUDDY_OPT} subgraph0.mlir -transform-preload-library="transform-library-paths=transform.mlir" -transform-interpreter="entry-point=codegen" | \
	${BUDDY_OPT} --pass-pipeline='builtin.module(func.func(nvgpu-optimize-shared-memory))' | \
	${BUDDY_OPT} -arith-expand -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -linalg-bufferize -convert-linalg-to-affine-loops -affine-loop-fusion -affine-parallelize -lower-affine -canonicalize -func-bufferize -arith-bufferize -tensor-bufferize -buffer-deallocation -finalizing-bufferize -canonicalize | \
	${BUDDY_OPT} -gpu-launch-sink-index-computations -canonicalize -legalize-shmem-outlining -canonicalize | \
	${BUDDY_OPT} -convert-memcpy-to-gpu -gpu-async-region -canonicalize | \
	${BUDDY_OPT} -convert-scf-to-cf -memref-expand -finalize-memref-to-llvm -convert-arith-to-llvm --convert-vector-to-llvm -convert-gpu-to-nvvm='has-redux=1' | \
	${BUDDY_OPT} -llvm-request-c-wrappers -canonicalize -cse -sccp | \
	${MLIR_OPT} --test-lower-to-nvvm="cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin" -o matmul-cubin.mlir

buddy-lenet-lower:
	@${BUDDY_OPT} ./fake-lenet.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" | \
	${BUDDY_OPT} \
		-eliminate-empty-tensors \
		-convert-tensor-to-linalg \
		-linalg-bufferize \
		-batchmatmul-optimize \
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
		-reconcile-unrealized-casts \
		-o ./log.mlir

buddy-lenet-translate:
	@${BUDDY_OPT} ./fake-lenet.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" | \
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
	${MLIR_TRANSLATE} -mlir-to-llvmir -o log.ll


buddy-lenet-run:
	@${BUDDY_OPT} ./fake-lenet.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" | \
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

buddy-lenet-opt-run:
	@${BUDDY_OPT} ./fake-lenet.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" | \
	${BUDDY_OPT} \
		-eliminate-empty-tensors \
		-convert-tensor-to-linalg \
		-linalg-bufferize \
		-batchmatmul-optimize \
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

