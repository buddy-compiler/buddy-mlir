#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-runner
LLI := ../../llvm/build/bin/lli
LLC := ../../llvm/build/bin/llc
OPT := ../../llvm/build/bin/opt
OPT_FLAG := -O3
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so

RISCV_GNU_TOOLCHAIN := ../../thirdparty/build-riscv-gnu-toolchain
RISCV_GNU_TOOLCHAIN_SYSROOT := ../../thirdparty/build-riscv-gnu-toolchain/sysroot
QEMU := ../../thirdparty/qemu/build/riscv64-linux-user/qemu-riscv64
LOCAL_CLANG := ../../thirdparty/build-local-clang/bin/clang
CROSS_LLI := ../../thirdparty/build-cross-clang/bin/lli
CROSS_MLIR_CPU_RUNNER := ../../thirdparty/build-cross-mlir/bin/mlir-runner
CROSS_MLIR_C_RUNNER_UTILS := ../../thirdparty/build-cross-mlir/lib/libmlir_c_runner_utils.so
CROSS_MLIR_RUNNER_UTILS := ../../thirdparty/build-cross-mlir/lib/libmlir_runner_utils.so
CROSS_MLIR_LIB := ../../thirdparty/build-cross-mlir/lib

MLIR_VECTOR_EXAMPLES := ../MLIRVector

.SECONDEXPANSION:
all-run: $$(run-targets)
