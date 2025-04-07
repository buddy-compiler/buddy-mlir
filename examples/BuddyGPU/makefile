#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-runner
LLC := ../../llvm/build/bin/llc

buddy-gpu-matmul-lower:
	@${BUDDY_OPT} matmul.mlir \
		-transform-preload-library="transform-library-paths=transform.mlir" \
		-transform-interpreter="entry-point=codegen" \
		-o log.mlir

buddy-gpu-matmul:
	@${BUDDY_OPT} matmul.mlir -transform-preload-library="transform-library-paths=transform.mlir" -transform-interpreter="entry-point=codegen" | \
	${BUDDY_OPT} --pass-pipeline='builtin.module(func.func(nvgpu-optimize-shared-memory))' | \
	${BUDDY_OPT} -arith-expand -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -linalg-bufferize -convert-linalg-to-affine-loops -affine-loop-fusion -affine-parallelize -lower-affine -canonicalize -func-bufferize -arith-bufferize -tensor-bufferize -buffer-deallocation -finalizing-bufferize -canonicalize | \
	${BUDDY_OPT} -gpu-launch-sink-index-computations -canonicalize -legalize-shmem-outlining -canonicalize | \
	${BUDDY_OPT} -convert-memcpy-to-gpu -gpu-async-region -canonicalize | \
	${BUDDY_OPT} -convert-scf-to-cf -memref-expand -finalize-memref-to-llvm -convert-arith-to-llvm --convert-vector-to-llvm -convert-gpu-to-nvvm='has-redux=1' | \
	${BUDDY_OPT} -llvm-request-c-wrappers -canonicalize -cse -sccp | \
	${MLIR_OPT} --test-lower-to-nvvm="cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin" -o matmul-cubin.mlir
