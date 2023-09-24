#!/bin/bash
BUDDY_OPT := ../../../build/bin/buddy-opt
MLIR_OPT := ../../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../../llvm/build/bin/mlir-cpu-runner
LLC := ../../../llvm/build/bin/llc
LLVM_AS := ../../../llvm/build/bin/llvm-as
OPT_FLAG := -O0

llama-lower:
	@${MLIR_OPT} ./llama.mlir \
		-pass-pipeline="builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		--arith-expand \
		--empty-tensor-to-alloc-tensor \
		--linalg-bufferize \
		--convert-linalg-to-loops \
		--func-bufferize \
		--arith-bufferize \
		--tensor-bufferize \
		--finalizing-bufferize \
		--convert-vector-to-scf \
		--convert-scf-to-cf \
		--expand-strided-metadata \
		--lower-affine \
		--convert-vector-to-llvm \
		--memref-expand \
		--arith-expand \
		--convert-arith-to-llvm \
		--finalize-memref-to-llvm \
		--convert-arith-to-llvm \
		--convert-math-to-llvm \
		--convert-math-to-libm \
		--llvm-request-c-wrappers \
		--convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} \
		-mlir-to-llvmir | \
	${LLVM_AS} | \
	${LLC} -filetype=obj --relocation-model=pic ${OPT_FLAG} -o llama.o

llama-batchmatmulopt:
	@${BUDDY_OPT} ./llama.mlir \
		-pass-pipeline="builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		--arith-expand \
		--empty-tensor-to-alloc-tensor \
		--tensor-bufferize \
		--linalg-bufferize \
		--batchmatmul-optimize \
		--verify-diagnostics \
		--expand-strided-metadata \
		--lower-affine \
		--convert-vector-to-llvm \
		--convert-linalg-to-loops \
		--func-bufferize \
		--arith-bufferize \
		--finalizing-bufferize \
		--convert-vector-to-scf \
		--convert-scf-to-cf \
		--memref-expand \
		--arith-expand \
		--convert-arith-to-llvm \
		--finalize-memref-to-llvm \
		--convert-arith-to-llvm \
		--convert-math-to-llvm \
		--convert-math-to-libm \
		--llvm-request-c-wrappers \
		--convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} \
		-mlir-to-llvmir | \
	${LLVM_AS} | \
	${LLC} -filetype=obj --relocation-model=pic ${OPT_FLAG} -o llama.o

llama-ompopt:
	@${MLIR_OPT} ./llama.mlir \
		-pass-pipeline="builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		--arith-expand \
		--eliminate-empty-tensors \
		--empty-tensor-to-alloc-tensor \
		--linalg-bufferize \
		--convert-linalg-to-affine-loops \
		--affine-loop-fusion \
		--affine-parallelize \
		--lower-affine \
		--convert-scf-to-openmp \
		--func-bufferize \
		--arith-bufferize \
		--tensor-bufferize \
		--buffer-deallocation \
		-o llama_linalg_lower.mlir
		--finalizing-bufferize \
		--convert-vector-to-scf \
		--expand-strided-metadata \
		--convert-vector-to-llvm \
		--memref-expand \
		--arith-expand \
		--convert-arith-to-llvm \
		--finalize-memref-to-llvm \
		--convert-scf-to-cf \
		--llvm-request-c-wrappers \
		--convert-openmp-to-llvm \
		--convert-arith-to-llvm \
		--convert-math-to-llvm \
		--convert-math-to-libm \
		--convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} \
		-mlir-to-llvmir | \
	${LLVM_AS} | \
	${LLC} -filetype=obj --relocation-model=pic ${OPT_FLAG} -o llama.o