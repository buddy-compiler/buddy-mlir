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

tensor-print-lower:
	@${MLIR_OPT} ./tensor-print.mlir \
		-arith-bufferize  -tensor-bufferize -linalg-bufferize \
		-func-bufferize -buffer-deallocation -convert-linalg-to-loops \
		-finalize-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts -o ./log.mlir

tensor-print-translate:
	@${MLIR_OPT} ./tensor-print.mlir \
		-arith-bufferize  -tensor-bufferize -linalg-bufferize \
		-func-bufferize -buffer-deallocation -convert-linalg-to-loops \
		-finalize-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

tensor-print-run:
	@${MLIR_OPT} ./tensor-print.mlir \
		-arith-bufferize  -tensor-bufferize -linalg-bufferize \
		-func-bufferize -buffer-deallocation -convert-linalg-to-loops \
		-finalize-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

tensor-collapse-shape-lower:
	@${MLIR_OPT} ./tensor-collapse-shape.mlir \
		-arith-bufferize -tensor-bufferize -func-bufferize \
		-finalizing-bufferize -buffer-deallocation \
		-expand-strided-metadata -lower-affine \
		-finalize-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts -o ./log.mlir

tensor-collapse-shape-translate:
	@${MLIR_OPT} ./tensor-collapse-shape.mlir \
		-arith-bufferize -tensor-bufferize -func-bufferize \
		-finalizing-bufferize -buffer-deallocation \
		-expand-strided-metadata -lower-affine \
		-finalize-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

tensor-collapse-shape-run:
	@${MLIR_OPT} ./tensor-collapse-shape.mlir \
		--one-shot-bufferize="bufferize-function-boundaries"  \
		-expand-strided-metadata -lower-affine \
		-finalize-memref-to-llvm -convert-func-to-llvm -convert-arith-to-llvm\
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

tensor-extract-lower:
	@${MLIR_OPT} ./tensor-extract.mlir \
		-arith-bufferize  -tensor-bufferize -linalg-bufferize -convert-vector-to-llvm \
		-func-bufferize -buffer-deallocation -convert-linalg-to-loops \
		 -finalize-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts -o ./log.mlir

tensor-extract-translate:
	@${MLIR_OPT} ./tensor-extract.mlir \
		-arith-bufferize  -tensor-bufferize -linalg-bufferize -convert-vector-to-llvm \
		-func-bufferize -buffer-deallocation -convert-linalg-to-loops \
		 -finalize-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

tensor-extract-run:
	@${MLIR_OPT} ./tensor-extract.mlir \
		--one-shot-bufferize="bufferize-function-boundaries" -convert-vector-to-llvm \
		-buffer-deallocation -convert-linalg-to-loops \
		 -finalize-memref-to-llvm -convert-func-to-llvm -convert-arith-to-llvm\
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

tensor-extract-slice-lower:
	@${MLIR_OPT} ./tensor-extract-slice.mlir \
		-arith-bufferize  -tensor-bufferize -linalg-bufferize -convert-vector-to-llvm \
		-func-bufferize -buffer-deallocation -convert-linalg-to-loops -expand-strided-metadata \
		 -finalize-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts -o ./log.mlir

tensor-extract-slice-translate:
	@${MLIR_OPT} ./tensor-extract-slice.mlir \
		-arith-bufferize  -tensor-bufferize -linalg-bufferize -convert-vector-to-llvm \
		-func-bufferize -buffer-deallocation -convert-linalg-to-loops -expand-strided-metadata \
		 -finalize-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

tensor-extract-slice-run:
	@${MLIR_OPT} ./tensor-extract-slice.mlir \
		--one-shot-bufferize="bufferize-function-boundaries" -convert-vector-to-llvm \
		 -convert-linalg-to-loops -convert-scf-to-cf -expand-strided-metadata  \
		 -finalize-memref-to-llvm -convert-func-to-llvm -convert-arith-to-llvm\
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

tensor-from-elements-lower:
	@${MLIR_OPT} ./tensor-from-elements.mlir \
		-arith-bufferize  -tensor-bufferize -linalg-bufferize -convert-vector-to-llvm \
		-func-bufferize -buffer-deallocation -convert-linalg-to-loops \
		 -finalize-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts -o ./log.mlir

tensor-from-elements-translate:
	@${MLIR_OPT} ./tensor-from-elements.mlir \
		-arith-bufferize  -tensor-bufferize -linalg-bufferize -convert-vector-to-llvm \
		-func-bufferize -buffer-deallocation -convert-linalg-to-loops \
		 -finalize-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

tensor-from-elements-run:
	@${MLIR_OPT} ./tensor-from-elements.mlir \
		--one-shot-bufferize="bufferize-function-boundaries" -convert-vector-to-llvm \
		-convert-linalg-to-loops \
		 -finalize-memref-to-llvm -convert-func-to-llvm -convert-arith-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

tensor-insert-lower:
	@${MLIR_OPT} ./tensor-insert.mlir \
		-arith-bufferize  -tensor-bufferize -linalg-bufferize -convert-vector-to-llvm \
		-func-bufferize -buffer-deallocation -convert-linalg-to-loops \
		 -finalize-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts -o ./log.mlir

tensor-insert-translate:
	@${MLIR_OPT} ./tensor-insert.mlir \
		-arith-bufferize  -tensor-bufferize -linalg-bufferize -convert-vector-to-llvm \
		-func-bufferize -buffer-deallocation -convert-linalg-to-loops \
		 -finalize-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

tensor-insert-run:
	@${MLIR_OPT} ./tensor-insert.mlir \
		-arith-bufferize  -tensor-bufferize -linalg-bufferize -convert-vector-to-llvm \
		-func-bufferize -buffer-deallocation -convert-linalg-to-loops \
		 -finalize-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

tensor-insert-slice-lower:
	@${MLIR_OPT} ./tensor-insert-slice.mlir \
		-arith-bufferize  -tensor-bufferize -linalg-bufferize -convert-vector-to-llvm \
		-func-bufferize -buffer-deallocation -convert-linalg-to-loops -expand-strided-metadata \
		 -finalize-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts -o ./log.mlir

tensor-insert-slice-translate:
	@${MLIR_OPT} ./tensor-insert-slice.mlir \
		-arith-bufferize  -tensor-bufferize -linalg-bufferize -convert-vector-to-llvm \
		-func-bufferize -buffer-deallocation -convert-linalg-to-loops -expand-strided-metadata \
		 -finalize-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

tensor-insert-slice-run:
	@${MLIR_OPT} ./tensor-insert-slice.mlir \
		-arith-bufferize  -tensor-bufferize -linalg-bufferize -convert-vector-to-llvm \
		-func-bufferize -buffer-deallocation -convert-linalg-to-loops -expand-strided-metadata \
		 -finalize-memref-to-llvm -convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
