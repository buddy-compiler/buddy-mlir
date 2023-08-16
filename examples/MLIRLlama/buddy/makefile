#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
LLC := ../../llvm/build/bin/llc
OPT_FLAG := -O0

resnet-18-lower:
	@${MLIR_OPT} ./ResNet-18.mlir \
		-pass-pipeline="builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		--test-linalg-transform-patterns="test-generalize-pad-tensor" \
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
		--convert-math-to-llvm \
		--llvm-request-c-wrappers \
		--convert-func-to-llvm \
		--reconcile-unrealized-casts \
		-o ./log.mlir

bert-lower:
	@${MLIR_OPT} ./test.mlir \
		-pass-pipeline="builtin.module(func.func(scf-bufferize),func.func(empty-tensor-to-alloc-tensor),func.func(linalg-bufferize),func-bufferize,arith-bufferize,func.func(tensor-bufferize),func.func(finalizing-bufferize),func.func(buffer-deallocation),func.func(convert-vector-to-scf{ full-unroll=1 }),convert-vector-to-llvm,finalize-memref-to-llvm,cse,func.func(lower-affine),func.func(arith-expand),func.func(convert-math-to-llvm),convert-math-to-libm,convert-linalg-to-llvm,expand-strided-metadata,finalize-memref-to-llvm,convert-scf-to-cf,convert-cf-to-llvm,cse,lower-affine,func.func(convert-arith-to-llvm),convert-func-to-llvm,canonicalize,convert-openmp-to-llvm,cse,reconcile-unrealized-casts)" \
		-o ./log.mlir
