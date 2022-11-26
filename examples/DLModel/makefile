#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
LLC := ../../llvm/build/bin/llc
OPT_FLAG := -O0

resnet-18-lower:
	@${MLIR_OPT} ./ResNet-18.mlir \
		-pass-pipeline="builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-tensor),func.func(tosa-to-linalg),func.func(tosa-to-arith))" | \
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
		--lower-affine \
		--convert-vector-to-llvm \
		-memref-expand -arith-expand \
		--convert-memref-to-llvm \
		--convert-math-to-llvm \
		--llvm-request-c-wrappers \
		--convert-func-to-llvm \
		--reconcile-unrealized-casts \
		-o ./log.mlir
