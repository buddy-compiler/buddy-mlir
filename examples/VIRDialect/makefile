#!/bin/bash
BUDDY_BUILD_DIR := ../../build/
LLVM_BUILD_DIR := ../../llvm/build/
BUDDY_OPT := ${BUDDY_BUILD_DIR}/bin/buddy-opt
MLIR_OPT := ${LLVM_BUILD_DIR}/bin/mlir-opt
MLIR_CPU_RUNNER := ${LLVM_BUILD_DIR}/bin/mlir-runner
LLC := ${LLVM_BUILD_DIR}/bin/llc
MLIR_TRANSLATE := ${LLVM_BUILD_DIR}/bin/mlir-translate
CLANG := ${LLVM_BUILD_DIR}/bin/clang
OPT_FLAG := -O3
MLIR_LIB := ${LLVM_BUILD_DIR}/lib/
RISCV_GNU_TOOLCHAIN := ${BUDDY_BUILD_DIR}/thirdparty/riscv-gnu-toolchain
CROSS_MLIR_BUILD_DIR := ../../llvm/build-cross-mlir-rv/
CROSS_MLIR_LIB := ${CROSS_MLIR_BUILD_DIR}/lib/

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.so
MLIR_ASYNC_RUNTIME := ${LLVM_BUILD_DIR}/lib/libmlir_async_runtime.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.dylib
MLIR_ASYNC_RUNTIME := ${LLVM_BUILD_DIR}/lib/libmlir_async_runtime.dylib
MTRIPLE := x86_64-apple-darwin
endif

vir-setvl-lower:
	${BUDDY_OPT} vir-setvl.mlir -o log.mlir

vir-dynamic-vec-type-lower:
	${BUDDY_OPT} vir-dynamic-vec-type.mlir \
	--allow-unregistered-dialect \
	-o log.mlir

vir-memory-lower:
	${BUDDY_OPT} vir-memory.mlir \
	-lower-vir-to-vector \
	-cse \
	-o log.mlir

vir-memory-run:
	${BUDDY_OPT} vir-memory.mlir \
		-lower-vir-to-vector \
		-cse \
		-convert-vector-to-scf \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vir-basic-lower:
	${BUDDY_OPT} vir-basic.mlir \
	-lower-vir-to-vector \
	-o log.mlir

vir-basic-run:
	${BUDDY_OPT} vir-basic.mlir \
		-lower-vir-to-vector \
		-cse \
		-convert-vector-to-scf \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vir-arith-lower:
	${BUDDY_OPT} vir-arith.mlir \
		-lower-vir-to-vector \
		-o log.mlir

vir-arith-run:
	${BUDDY_OPT} vir-arith.mlir \
		-lower-vir-to-vector \
		-cse \
		-convert-vector-to-scf \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

# Default values for STEP and SIZE
STEP ?= 4
SIZE ?= 4096

vector-saxpy-fixed-run:
	sed 's/STEP_PLACEHOLDER/${STEP}/g;s/SIZE_PLACEHOLDER/${SIZE}/g' vector-saxpy-fixed.mlir | \
	${BUDDY_OPT} \
		-convert-vector-to-scf \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

vector-saxpy-fixed-aot:
	sed 's/STEP_PLACEHOLDER/${STEP}/g;s/SIZE_PLACEHOLDER/${SIZE}/g' vector-saxpy-fixed.mlir | \
	${BUDDY_OPT} \
		-convert-vector-to-scf \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir -o log.ll
	${CLANG} log.ll ${OPT_FLAG} \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${MLIR_LIB} ./a.out || true

vector-saxpy-fixed-cross-rvv-aot:
	sed 's/STEP_PLACEHOLDER/${STEP}/g;s/SIZE_PLACEHOLDER/${SIZE}/g' vector-saxpy-fixed.mlir | \
	${BUDDY_OPT} \
		-convert-vector-to-scf \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir -o log.ll
	${CLANG} -O3 log.ll \
		-march=rv64gcv --target=riscv64-unknown-linux-gnu -fPIC \
		--sysroot=${RISCV_GNU_TOOLCHAIN}/sysroot \
		--gcc-toolchain=${RISCV_GNU_TOOLCHAIN} \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o saxpy.fixed.${STEP}.${SIZE}.out

vector-saxpy-fixed-cross-rvv-asm:
	sed 's/STEP_PLACEHOLDER/${STEP}/g;s/SIZE_PLACEHOLDER/${SIZE}/g' vector-saxpy-fixed.mlir | \
	${BUDDY_OPT} \
		-convert-vector-to-scf \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir -o log.ll
	${CLANG} -O3 log.ll \
		-march=rv64gcv --target=riscv64-unknown-linux-gnu -fPIC \
		--sysroot=${RISCV_GNU_TOOLCHAIN}/sysroot \
		--gcc-toolchain=${RISCV_GNU_TOOLCHAIN} \
		-fno-inline -S \
		-o saxpy.fixed.${STEP}.${SIZE}.s

vector-saxpy-scalable-cross-rvv-aot:
	sed 's/STEP_PLACEHOLDER/${STEP}/g;s/SIZE_PLACEHOLDER/${SIZE}/g' vector-saxpy-scalable.mlir | \
	${BUDDY_OPT} \
		-convert-vector-to-scf \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir -o log.ll
	${CLANG} -O3 log.ll \
		-march=rv64gcv --target=riscv64-unknown-linux-gnu -fPIC \
		--sysroot=${RISCV_GNU_TOOLCHAIN}/sysroot \
		--gcc-toolchain=${RISCV_GNU_TOOLCHAIN} \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
		-o saxpy.scalable.${STEP}.${SIZE}.out

vector-saxpy-scalable-cross-rvv-asm:
	sed 's/STEP_PLACEHOLDER/${STEP}/g;s/SIZE_PLACEHOLDER/${SIZE}/g' vector-saxpy-scalable.mlir | \
	${BUDDY_OPT} \
		-convert-vector-to-scf \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir -o log.ll
	${CLANG} -O3 log.ll \
		-march=rv64gcv --target=riscv64-unknown-linux-gnu -fPIC \
		--sysroot=${RISCV_GNU_TOOLCHAIN}/sysroot \
		--gcc-toolchain=${RISCV_GNU_TOOLCHAIN} \
		-fno-inline -S \
		-o saxpy.scalable.${STEP}.${SIZE}.s

vector-saxpy-vp-cross-rvv-aot:
	sed 's/STEP_PLACEHOLDER/${STEP}/g;s/SIZE_PLACEHOLDER/${SIZE}/g' vector-saxpy-vp.mlir | \
	${BUDDY_OPT} \
		-lower-vector-exp \
		-convert-vector-to-scf \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-convert-index-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir -o log.ll
	${CLANG} -O3 log.ll \
		-march=rv64gcv --target=riscv64-unknown-linux-gnu -fPIC \
		--sysroot=${RISCV_GNU_TOOLCHAIN}/sysroot \
		--gcc-toolchain=${RISCV_GNU_TOOLCHAIN} \
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils \
			-o saxpy.vp.${STEP}.${SIZE}.out

vector-saxpy-vp-cross-rvv-asm:
	sed 's/STEP_PLACEHOLDER/${STEP}/g;s/SIZE_PLACEHOLDER/${SIZE}/g' vector-saxpy-vp.mlir | \
	${BUDDY_OPT} \
		-lower-vector-exp \
		-convert-vector-to-scf \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-convert-index-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir -o log.ll
	${CLANG} -O3 log.ll \
		-march=rv64gcv --target=riscv64-unknown-linux-gnu -fPIC \
		--sysroot=${RISCV_GNU_TOOLCHAIN}/sysroot \
		--gcc-toolchain=${RISCV_GNU_TOOLCHAIN} \
		-fno-inline -S \
		-o saxpy.vp.${STEP}.${SIZE}.s
