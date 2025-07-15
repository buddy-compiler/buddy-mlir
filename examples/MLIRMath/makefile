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
MLIR_ASYNC_RUNTIME := ../../llvm/build/lib/libmlir_async_runtime.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
MLIR_ASYNC_RUNTIME := ./../llvm/build/lib/libmlir_async_runtime.dylib
MTRIPLE := x86_64-apple-darwin
endif

math-atan2-lower:
	@${MLIR_OPT} ./math-atan2.mlir \
		-convert-math-to-llvm -convert-vector-to-llvm -convert-func-to-llvm -convert-arith-to-llvm \
		-reconcile-unrealized-casts \
		-o ./log.mlir

math-atan2-translate:
	@${MLIR_OPT} ./math-atan2.mlir \
		-convert-math-to-llvm -convert-vector-to-llvm -convert-func-to-llvm -convert-arith-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

math-atan2-run:
	@${MLIR_OPT} ./math-atan2.mlir \
		-convert-math-to-llvm -convert-vector-to-llvm -convert-func-to-llvm -convert-arith-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

math-rsqrt-lower:
	@${MLIR_OPT} ./math-rsqrt.mlir \
		-convert-math-to-llvm -convert-func-to-llvm -convert-arith-to-llvm \
		-o ./log.mlir

math-rsqrt-translate:
	@${MLIR_OPT} ./math-rsqrt.mlir \
		-convert-math-to-llvm -convert-func-to-llvm -convert-arith-to-llvm | \
	${MLIR_TRANSLATE} \
		-mlir-to-llvmir -o log.ll

math-rsqrt-lower-x86:
	@${MLIR_OPT} ./math-rsqrt.mlir \
		-test-math-polynomial-approximation=enable-avx2 \
		-convert-vector-to-llvm="enable-x86vector" \
		-convert-math-to-llvm -convert-func-to-llvm \
		-o ./log.mlir

math-rsqrt-translate-x86:
	@${MLIR_OPT} ./math-rsqrt.mlir \
		-test-math-polynomial-approximation=enable-avx2 \
		-convert-vector-to-llvm="enable-x86vector" \
		-convert-math-to-llvm -convert-func-to-llvm -convert-arith-to-llvm | \
	${MLIR_TRANSLATE} \
		-mlir-to-llvmir -o log.ll

math-rsqrt-asm-x86:
	@${MLIR_OPT} ./math-rsqrt.mlir \
		-test-math-polynomial-approximation=enable-avx2 \
		-convert-vector-to-llvm="enable-x86vector" \
		-convert-math-to-llvm -convert-func-to-llvm -convert-arith-to-llvm | \
	${MLIR_TRANSLATE} \
		-mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f \
		--filetype=asm -o log.s

math-rsqrt-run:
	@${MLIR_OPT} ./math-rsqrt.mlir \
		-convert-vector-to-scf -convert-scf-to-cf -convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-convert-arith-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
