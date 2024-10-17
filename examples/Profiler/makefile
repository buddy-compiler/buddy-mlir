#!/bin/bash
BUDDY_BUILD_DIR := ../../build/
LLVM_BUILD_DIR := ../../llvm/build
BUDDY_OPT := ${BUDDY_BUILD_DIR}/bin/buddy-opt
MLIR_OPT := ${LLVM_BUILD_DIR}/bin/mlir-opt
MLIR_TRANSLATE := ${LLVM_BUILD_DIR}/bin/mlir-translate
MLIR_CPU_RUNNER := ${LLVM_BUILD_DIR}/bin/mlir-cpu-runner
LLC := ${LLVM_BUILD_DIR}/bin/llc
OPT_FLAG := -O0
LLVM_TOOLS_BINARY_DIR = ${LLVM_BUILD_DIR}/bin
BUDDY_LIB_DIR = ${BUDDY_BUILD_DIR}/lib

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.dylib
MTRIPLE := x86_64-apple-darwin
endif

buddy-profier-run:
	@${BUDDY_OPT} ./linalg-batchmatmul-f32.mlir \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-expand-strided-metadata \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
	

buddy-forward-gen:
	${BUDDY_OPT} ./forward.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), \
		empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, \
		func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" | \
	${BUDDY_OPT} \
		-pass-pipeline "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), \
		eliminate-empty-tensors, func.func(llvm-request-c-wrappers), \
		convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf, \
		convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, \
		convert-func-to-llvm, reconcile-unrealized-casts)" | \
		${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir | \
		${LLVM_TOOLS_BINARY_DIR}/llvm-as | \
		${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O3 \
		-o forward.o
		${LLVM_TOOLS_BINARY_DIR}/clang -shared -o libforward.so forward.o subgraph0.o ${BUDDY_LIB_DIR}/libStaticMLIRCRunnerUtils.a ${BUDDY_LIB_DIR}/libStaticMLIRRunnerUtils.a


buddy-subgraph0-gen:
	${BUDDY_OPT} ./subgraph0.mlir \
		-pass-pipeline \
		"builtin.module(func.func(tosa-to-linalg-named, tosa-to-arith, tosa-to-linalg, tosa-to-tensor))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-func-bufferize-dynamic-offset \
		-arith-bufferize \
		-func-bufferize \
		-tensor-bufferize \
		-linalg-bufferize \
		-finalizing-bufferize \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-llvm-request-c-wrappers \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-expand-strided-metadata \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir | \
	${LLVM_TOOLS_BINARY_DIR}/llvm-as | \
	${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3 \
		-o subgraph0.o
	${LLVM_TOOLS_BINARY_DIR}/clang -shared -o libsubgraph0.so subgraph0.o  ${BUDDY_LIB_DIR}/libStaticMLIRCRunnerUtils.a ${BUDDY_LIB_DIR}/libStaticMLIRRunnerUtils.a

clean:
	rm -f forward.o subgraph0.o libforward.so libsubgraph0.so




