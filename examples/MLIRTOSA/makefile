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
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
MTRIPLE := x86_64-apple-darwin
endif

tosa-resize-lower-to-linalg:
	@${MLIR_OPT} ./tosa-resize.mlir \
		-pass-pipeline="builtin.module( \
			func.func(tosa-to-linalg) \
		)" \
		-o ./log.mlir

tosa-resize-lower:
	@${MLIR_OPT} ./tosa-resize.mlir \
		-pass-pipeline="builtin.module( \
			func.func(tosa-to-linalg), \
			empty-tensor-to-alloc-tensor, \
			arith-bufferize, \
			func.func(tensor-bufferize, linalg-bufferize), \
			func-bufferize, \
			func.func(buffer-deallocation, convert-linalg-to-loops), \
			convert-scf-to-cf, \
			finalize-memref-to-llvm, \
			convert-math-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
		)" \
		-o ./log.mlir

tosa-resize-translate:
	@${MLIR_OPT} ./tosa-resize.mlir \
		-pass-pipeline="builtin.module( \
			func.func(tosa-to-linalg), \
			empty-tensor-to-alloc-tensor, \
			arith-bufferize, \
			func.func(tensor-bufferize, linalg-bufferize), \
			func-bufferize, \
			func.func(buffer-deallocation, convert-linalg-to-loops), \
			convert-scf-to-cf, \
			finalize-memref-to-llvm, \
			convert-math-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
		)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

tosa-resize-run:
	@${MLIR_OPT} ./tosa-resize.mlir \
		-pass-pipeline="builtin.module( \
			func.func(tosa-to-linalg), \
			empty-tensor-to-alloc-tensor, \
			arith-bufferize, \
			func.func(tensor-bufferize, linalg-bufferize), \
			func-bufferize, \
			func.func(buffer-deallocation, convert-linalg-to-loops), \
			convert-scf-to-cf, \
			finalize-memref-to-llvm, \
			convert-math-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
		)" | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

tosa-sigmoid-lower:
	@${MLIR_OPT} ./tosa-sigmoid.mlir \
		-pass-pipeline="builtin.module( \
			func.func(tosa-to-linalg), \
			empty-tensor-to-alloc-tensor, \
			arith-bufferize, \
			func.func(tensor-bufferize, linalg-bufferize), \
			func-bufferize, \
			func.func(buffer-deallocation, convert-linalg-to-loops), \
			convert-scf-to-cf, \
			finalize-memref-to-llvm, \
			convert-math-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
		)" \
		-o ./log.mlir

tosa-sigmoid-translate:
	@${MLIR_OPT} ./tosa-sigmoid.mlir \
		-pass-pipeline="builtin.module( \
			func.func(tosa-to-linalg), \
			empty-tensor-to-alloc-tensor, \
			arith-bufferize, \
			func.func(tensor-bufferize, linalg-bufferize), \
			func-bufferize, \
			func.func(buffer-deallocation, convert-linalg-to-loops), \
			convert-scf-to-cf, \
			finalize-memref-to-llvm, \
			convert-math-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
		)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

tosa-sigmoid-run:
	@${MLIR_OPT} ./tosa-sigmoid.mlir \
		-pass-pipeline="builtin.module( \
			func.func(tosa-to-linalg), \
			empty-tensor-to-alloc-tensor, \
			arith-bufferize, \
			func.func(tensor-bufferize, linalg-bufferize), \
			func-bufferize, \
			func.func(buffer-deallocation, convert-linalg-to-loops), \
			convert-scf-to-cf, \
			finalize-memref-to-llvm, \
			convert-math-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
		)" | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

tosa-log-lower:
	@${MLIR_OPT} ./tosa-log.mlir \
		-pass-pipeline="builtin.module( \
			func.func(tosa-to-linalg), \
			empty-tensor-to-alloc-tensor, \
			arith-bufferize, \
			func.func(tensor-bufferize, linalg-bufferize), \
			func-bufferize, \
			func.func(buffer-deallocation, convert-linalg-to-loops), \
			convert-scf-to-cf, \
			finalize-memref-to-llvm, \
			convert-math-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
		)" \
		-o ./log.mlir

tosa-log-translate:
	@${MLIR_OPT} ./tosa-log.mlir \
		-pass-pipeline="builtin.module( \
			func.func(tosa-to-linalg), \
			empty-tensor-to-alloc-tensor, \
			arith-bufferize, \
			func.func(tensor-bufferize, linalg-bufferize), \
			func-bufferize, \
			func.func(buffer-deallocation, convert-linalg-to-loops), \
			convert-scf-to-cf, \
			finalize-memref-to-llvm, \
			convert-math-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
		)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

tosa-log-run:
	@${MLIR_OPT} ./tosa-log.mlir \
		-pass-pipeline="builtin.module( \
			func.func(tosa-to-linalg), \
			empty-tensor-to-alloc-tensor, \
			arith-bufferize, \
			func.func(tensor-bufferize, linalg-bufferize), \
			func-bufferize, \
			func.func(buffer-deallocation, convert-linalg-to-loops), \
			convert-scf-to-cf, \
			finalize-memref-to-llvm, \
			convert-math-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
		)" | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

tosa-add-lower:
	@${MLIR_OPT} ./tosa-add.mlir \
		-pass-pipeline="builtin.module( \
			func.func(tosa-to-linalg, tosa-to-tensor), \
			empty-tensor-to-alloc-tensor, \
			arith-bufferize, \
			func.func(tensor-bufferize, linalg-bufferize), \
			func-bufferize, \
			func.func(buffer-deallocation, convert-linalg-to-loops), \
			convert-scf-to-cf, \
			finalize-memref-to-llvm, \
			convert-math-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
		)" \
		-o ./log.mlir

tosa-add-translate:
	@${MLIR_OPT} ./tosa-add.mlir \
		-pass-pipeline="builtin.module( \
			func.func(tosa-to-linalg, tosa-to-tensor), \
			empty-tensor-to-alloc-tensor, \
			arith-bufferize, \
			func.func(tensor-bufferize, linalg-bufferize), \
			func-bufferize, \
			func.func(buffer-deallocation, convert-linalg-to-loops), \
			convert-scf-to-cf, \
			finalize-memref-to-llvm, \
			convert-math-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
		)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

tosa-add-run:
	@${MLIR_OPT} ./tosa-add.mlir \
		-pass-pipeline="builtin.module( \
			func.func(tosa-to-linalg, tosa-to-tensor), \
			empty-tensor-to-alloc-tensor, \
			arith-bufferize, \
			func.func(tensor-bufferize, linalg-bufferize), \
			func-bufferize, \
			func.func(buffer-deallocation, convert-linalg-to-loops), \
			convert-scf-to-cf, \
			finalize-memref-to-llvm, \
			convert-math-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
		)" | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

tosa-concat-lower:
	@${MLIR_OPT} ./tosa-concat.mlir \
		-pass-pipeline="builtin.module( \
			func.func(tosa-to-linalg, tosa-to-tensor), \
			empty-tensor-to-alloc-tensor, \
			arith-bufferize, \
			func.func(tensor-bufferize, linalg-bufferize), \
			func-bufferize, \
			func.func(buffer-deallocation, convert-linalg-to-loops), \
			convert-scf-to-cf, \
			expand-strided-metadata, \
			finalize-memref-to-llvm, \
			convert-math-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
		)" \
		-o ./log.mlir

tosa-concat-translate:
	@${MLIR_OPT} ./tosa-concat.mlir \
		-pass-pipeline="builtin.module( \
			func.func(tosa-to-linalg, tosa-to-tensor), \
			empty-tensor-to-alloc-tensor, \
			arith-bufferize, \
			func.func(tensor-bufferize, linalg-bufferize), \
			func-bufferize, \
			func.func(buffer-deallocation, convert-linalg-to-loops), \
			convert-scf-to-cf, \
			expand-strided-metadata, \
			finalize-memref-to-llvm, \
			convert-math-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
		)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

tosa-concat-run:
	@${MLIR_OPT} ./tosa-concat.mlir \
		-pass-pipeline="builtin.module( \
			func.func(tosa-to-linalg, tosa-to-tensor), \
			empty-tensor-to-alloc-tensor, \
			arith-bufferize, \
			func.func(tensor-bufferize, linalg-bufferize), \
			func-bufferize, \
			func.func(buffer-deallocation, convert-linalg-to-loops), \
			convert-scf-to-cf, \
			expand-strided-metadata, \
			finalize-memref-to-llvm, \
			convert-math-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
		)" | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
