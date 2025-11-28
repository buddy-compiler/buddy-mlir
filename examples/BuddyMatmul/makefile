#!/bin/bash
BUDDY_BUILD_DIR := ../../build/
LLVM_BUILD_DIR := ../../llvm/build/
BUDDY_OPT := ${BUDDY_BUILD_DIR}/bin/buddy-opt
MLIR_OPT := ${LLVM_BUILD_DIR}/bin/mlir-opt
MLIR_TRANSLATE := ${LLVM_BUILD_DIR}/bin/mlir-translate
MLIR_CPU_RUNNER := ${LLVM_BUILD_DIR}/bin/mlir-runner
LLC := ${LLVM_BUILD_DIR}/bin/llc
OPT_FLAG := -O0

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.so
LIB_OMP := ${LLVM_BUILD_DIR}/lib/libomp.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.dylib
MTRIPLE := x86_64-apple-darwin
endif

linalg-batchmatmul-f32-run:
	@${BUDDY_OPT} ./linalg-batchmatmul-f32.mlir \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
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

linalg-batchmatmul-vectorization-lower:
	@${BUDDY_OPT} ./linalg-batchmatmul-f32.mlir \
		-batchmatmul-optimize \
		-o ./log.mlir

linalg-batchmatmul-f32-vectorization-run:
	@${BUDDY_OPT} ./linalg-batchmatmul-f32.mlir \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
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

linalg-batchmatmul-f32-omp-lower:
	@${BUDDY_OPT} ./linalg-batchmatmul-f32.mlir \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm  \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts  \
		-o log.mlir

linalg-batchmatmul-f32-omp-run:
	@${BUDDY_OPT} ./linalg-batchmatmul-f32.mlir \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm  \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS} \
		-shared-libs=${LIB_OMP}

linalg-matmul-transpose-b-f32-run:
	@${BUDDY_OPT} ./linalg-transpose-matmul-b-f32.mlir\
		-matmul-transpose-b-vectorization \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
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

linalg-batchmatmul-transpose-b-f32-run:
	@${BUDDY_OPT} ./linalg-transpose-batchmatmul-b.mlir\
		-batchmatmul-transpose-b-vectorization \
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

batchmatmul-transpose-b-vec-run:
	@${BUDDY_OPT} ./transpose-batchmatmul-b-vec.mlir\
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

batchmatmul-vectorization-run:
	@${BUDDY_OPT} ./batchmatmul-vectorization.mlir \
		-convert-linalg-to-loops \
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

# AMX BF16 test
# AMX requires kernel-level permissions that cannot be obtained in JIT environments
amx-bf16-matmul-aot:
	@echo "AMX JIT version not supported due to permission issues."
	@echo "Running simplified AOT version instead..."
	@${BUDDY_OPT} ./amx-bf16-matmul.mlir \
		--llvm-request-c-wrappers \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-vector-to-llvm="enable-amx" \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o amx-quick.ll
	@${LLC} -mtriple ${MTRIPLE} -mattr=+amx-bf16,+amx-tile,+amx-int8 ${OPT_FLAG} -filetype=obj amx-quick.ll -o amx-quick.o
	@gcc -c amx-wrapper.c -o amx-wrapper-quick.o
	@g++ amx-wrapper-quick.o amx-quick.o -o amx-quick -L${LLVM_BUILD_DIR}/lib -lmlir_runner_utils -lmlir_c_runner_utils -lmlir_float16_utils -lpthread -Wl,-rpath,${LLVM_BUILD_DIR}/lib
	@echo "Running AMX quick test..."
	@LD_LIBRARY_PATH=${LLVM_BUILD_DIR}/lib ./amx-quick
	@rm -f amx-quick.ll amx-quick.o amx-wrapper-quick.o amx-quick

# AMX INT8 test
# AMX requires kernel-level permissions that cannot be obtained in JIT environments
amx-int8-matmul-aot:
	@echo "AMX INT8 JIT version not supported due to permission issues."
	@echo "Running simplified AOT version instead..."
	@${BUDDY_OPT} ./amx-int8-matmul.mlir \
		--llvm-request-c-wrappers \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-vector-to-llvm="enable-amx" \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o amx-int8-quick.ll
	@${LLC} -mtriple ${MTRIPLE} -mattr=+amx-bf16,+amx-tile,+amx-int8 ${OPT_FLAG} -filetype=obj amx-int8-quick.ll -o amx-int8-quick.o
	@gcc -c amx-wrapper.c -o amx-wrapper-int8-quick.o
	@g++ amx-wrapper-int8-quick.o amx-int8-quick.o -o amx-int8-quick -L${LLVM_BUILD_DIR}/lib -lmlir_runner_utils -lmlir_c_runner_utils -lpthread -Wl,-rpath,${LLVM_BUILD_DIR}/lib
	@echo "Running AMX INT8 quick test..."
	@LD_LIBRARY_PATH=${LLVM_BUILD_DIR}/lib ./amx-int8-quick
	@rm -f amx-int8-quick.ll amx-int8-quick.o amx-wrapper-int8-quick.o amx-int8-quick

# Check what AMX INT8 instructions are generated
amx-int8-matmul-asm:
	@${BUDDY_OPT} ./amx-int8-matmul.mlir \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-vector-to-llvm="enable-amx" \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} -mtriple ${MTRIPLE} -mattr=+amx-bf16,+amx-tile,+amx-int8 ${OPT_FLAG} -filetype=asm -o amx-int8-asm.s
	@echo "Generated assembly in amx-int8-asm.s"

# Check what AMX instructions are generated
amx-bf16-matmul-asm:
	@${BUDDY_OPT} ./amx-bf16-matmul.mlir \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-vector-to-llvm="enable-amx" \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} -mtriple ${MTRIPLE} -mattr=+amx-bf16,+amx-tile,+amx-int8 ${OPT_FLAG} -filetype=asm -o amx-asm.s
	@echo "Generated assembly in amx-asm.s"

linalg-bf16-matmul-lower:
	@${BUDDY_OPT} ./linalg-bf16-matmul.mlir \
		-convert-linalg-to-loops \
		-o log.mlir

linalg-bf16-matmul-run:
	@echo "Compiling and running linalg-to-AMX example..."
	@${BUDDY_OPT} ./linalg-to-amx-example.mlir \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-llvm-request-c-wrappers \
		-convert-vector-to-llvm="enable-amx" \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-cf-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o linalg-amx.ll
	@${LLC} -mtriple=x86_64-unknown-linux-gnu -mattr=+amx-bf16,+amx-tile,+amx-int8 \
		-relocation-model=pic linalg-amx.ll -filetype=obj -o linalg-amx.o
	@gcc -c -fPIC amx-wrapper.c -o amx-wrapper.o
	@g++ -no-pie linalg-amx.o amx-wrapper.o -o linalg-amx-exec \
		-L${LLVM_BUILD_DIR}/lib -lmlir_runner_utils -lmlir_c_runner_utils -lmlir_float16_utils -lpthread \
		-Wl,-rpath,${LLVM_BUILD_DIR}/lib
	@echo "Running executable..."
	@./linalg-amx-exec
	@rm -f linalg-amx.ll linalg-amx.o amx-wrapper.o linalg-amx-exec

linalg-int8-matmul-lower:
	@${BUDDY_OPT} ./test-amx-int8-pass.mlir \
		-matmul-amx-int8 \
		-o log.mlir

linalg-int8-matmul-run:
	@echo "Compiling and running linalg-to-AMX INT8 example..."
	@${BUDDY_OPT} ./test-amx-int8-pass.mlir \
		-matmul-amx-int8 \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-llvm-request-c-wrappers \
		-convert-vector-to-llvm="enable-amx" \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-cf-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o linalg-amx-int8.ll
	@${LLC} -mtriple=x86_64-unknown-linux-gnu -mattr=+amx-bf16,+amx-tile,+amx-int8 \
		-relocation-model=pic linalg-amx-int8.ll -filetype=obj -o linalg-amx-int8.o
	@gcc -c -fPIC amx-wrapper.c -o amx-wrapper-int8.o
	@g++ -no-pie linalg-amx-int8.o amx-wrapper-int8.o -o linalg-amx-int8-exec \
		-L${LLVM_BUILD_DIR}/lib -lmlir_runner_utils -lmlir_c_runner_utils -lpthread \
		-Wl,-rpath,${LLVM_BUILD_DIR}/lib
	@echo "Running executable..."
	@./linalg-amx-int8-exec
	@rm -f linalg-amx-int8.ll linalg-amx-int8.o amx-wrapper-int8.o linalg-amx-int8-exec

# Check what AMX INT8 instructions are generated by -matmul-amx-int8 pass
linalg-int8-matmul-asm:
	@${BUDDY_OPT} ./test-amx-int8-pass.mlir \
		-matmul-amx-int8 \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-vector-to-llvm="enable-amx" \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} -mtriple ${MTRIPLE} -mattr=+amx-bf16,+amx-tile,+amx-int8 ${OPT_FLAG} -filetype=asm -o linalg-amx-int8.s
	@echo "Generated assembly in linalg-amx-int8.s"
