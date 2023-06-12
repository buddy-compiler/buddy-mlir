#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
LLC := ../../llvm/build/bin/llc
OPT_FLAG := -O0

RISCV_GNU_TOOLCHAIN := ../../thirdparty/build-riscv-gnu-toolchain
RISCV_GNU_TOOLCHAIN_SYSROOT := ../../thirdparty/build-riscv-gnu-toolchain/sysroot
QEMU := ../../thirdparty/qemu/build/riscv64-linux-user/qemu-riscv64
LOCAL_CLANG := ../../thirdparty/build-local-clang/bin/clang
CROSS_LLI := ../../thirdparty/build-cross-clang/bin/lli
CROSS_MLIR_CPU_RUNNER := ../../thirdparty/build-cross-mlir/bin/mlir-cpu-runner
CROSS_MLIR_C_RUNNER_UTILS := ../../thirdparty/build-cross-mlir/lib/libmlir_c_runner_utils.so
CROSS_MLIR_RUNNER_UTILS := ../../thirdparty/build-cross-mlir/lib/libmlir_runner_utils.so
CROSS_MLIR_LIB := ../../thirdparty/build-cross-mlir/lib

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

bert-stat:
	@${BUDDY_OPT} ./bert.mlir -print-op-stats  > /dev/null

bert-lower:
	@${MLIR_OPT} ./bert.mlir \
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
		--test-math-polynomial-approximation \
		--convert-math-to-llvm \
		--llvm-request-c-wrappers \
		--convert-func-to-llvm \
		--reconcile-unrealized-casts \
		-o ./log.mlir

sentiment-classification-run-rv: bert-rvv.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++ -c sentiment-classification.cpp
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++ bert-rvv.o sentiment-classification.o \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -no-pie -mabi=lp64d -fPIC -o sentiment-classification
	@LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 sentiment-classification

bert-rvv.o:
	@${MLIR_OPT} ./bert-with-weights.mlir \
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
		-test-math-polynomial-approximation \
		--convert-math-to-llvm \
		--llvm-request-c-wrappers \
		--convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} -O3 -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -filetype=obj -riscv-v-vector-bits-min=128 \
		-o ./bert-rvv.o

clean:
	@rm -rf *.o sentiment-classification
