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
MLIR_ASYNC_RUNTIME := ../../llvm/build/lib/libmlir_async_runtime.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
MLIR_ASYNC_RUNTIME := ./../llvm/build/lib/libmlir_async_runtime.dylib
MTRIPLE := x86_64-apple-darwin
endif

scf-while-lower:
	@${MLIR_OPT} ./scf-while.mlir \
		--convert-scf-to-cf --convert-vector-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

scf-while-translate:
	@${MLIR_OPT} ./scf-while.mlir \
		--convert-scf-to-cf --convert-vector-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

scf-while-run:
	@${MLIR_OPT} ./scf-while.mlir \
		--convert-scf-to-cf --convert-vector-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

scf-parallel-lower:
	@${MLIR_OPT} ./scf-parallel.mlir \
	    -async-parallel-for \
		-o ./log.mlir

scf-parallel-run:
	@${MLIR_OPT} ./scf-parallel.mlir \
	    -async-parallel-for \
        -async-to-async-runtime \
		-async-runtime-ref-counting \
        -async-runtime-ref-counting-opt \
		-arith-expand \
        -convert-async-to-llvm \
        -convert-scf-to-cf \
        -convert-memref-to-llvm \
        -convert-func-to-llvm \
        -reconcile-unrealized-casts |\
    ${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
	    -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_ASYNC_RUNTIME}
