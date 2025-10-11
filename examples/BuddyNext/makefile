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

# Platform detection
UNAME_M := $(shell uname -m)
ifeq ($(findstring riscv,$(UNAME_M)),riscv)
MARCH_FLAG := -march=rv64gcv
else
MARCH_FLAG := -march=native
endif

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.so
OMP_LIB := ${LLVM_BUILD_DIR}/lib/libomp.so
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.dylib
OMP_LIB := ${LLVM_BUILD_DIR}/lib/libomp.dylib
endif

# ==============================================================================
# Evaluation of Modules in DeepSeek-R1 Distill Qwen 1.5B
# ==============================================================================

next-test-lower:
	@${MLIR_OPT} ./next-embedding.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-o log.mlir

next-embedding-run:
	@${MLIR_OPT} ./next-embedding.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-convert-elementwise-to-linalg \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-expand-strided-metadata \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-matmul-parallel-vectorization-optimize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-convert-vector-to-scf \
		-lower-affine \
		-convert-scf-to-openmp \
		-func-bufferize-dynamic-offset \
		-cse \
		-memref-expand \
		-arith-expand \
		-convert-vector-to-llvm \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS} \
		-shared-libs=${OMP_LIB}

next-embedding-aot-omp:
	@${MLIR_OPT} ./next-embedding.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-convert-elementwise-to-linalg \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-expand-strided-metadata \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-matmul-parallel-vectorization-optimize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-convert-vector-to-scf \
		-lower-affine \
		-convert-scf-to-openmp \
		-func-bufferize-dynamic-offset \
		-cse \
		-memref-expand \
		-arith-expand \
		-convert-vector-to-llvm \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-embedding.out
	./next-embedding.out || true

next-positional-encoding-aot-omp:
	@${MLIR_OPT} ./next-positional-encoding.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-convert-elementwise-to-linalg \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-expand-strided-metadata \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-matmul-parallel-vectorization-optimize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-convert-vector-to-scf \
		-lower-affine \
		-convert-scf-to-openmp \
		-func-bufferize-dynamic-offset \
		-cse \
		-memref-expand \
		-arith-expand \
		-convert-vector-to-llvm \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-positional-encoding.out
	./next-positional-encoding.out || true

next-positional-encoding-aot:
	@${MLIR_OPT} ./next-positional-encoding.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-convert-elementwise-to-linalg \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-expand-strided-metadata \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-matmul-parallel-vectorization-optimize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-convert-vector-to-scf \
		-lower-affine \
		-func-bufferize-dynamic-offset \
		-cse \
		-memref-expand \
		-arith-expand \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-positional-encoding.out
	./next-positional-encoding.out || true

next-norm-aot-omp:
	@${MLIR_OPT} ./next-norm.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-convert-elementwise-to-linalg \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-expand-strided-metadata \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-matmul-parallel-vectorization-optimize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-convert-vector-to-scf \
		-lower-affine \
		-convert-scf-to-openmp \
		-func-bufferize-dynamic-offset \
		-cse \
		-memref-expand \
		-arith-expand \
		-convert-vector-to-llvm \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-norm.out
	./next-norm.out || true

next-norm-aot:
	@${MLIR_OPT} ./next-norm.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-convert-elementwise-to-linalg \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-expand-strided-metadata \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-matmul-parallel-vectorization-optimize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-convert-vector-to-scf \
		-lower-affine \
		-func-bufferize-dynamic-offset \
		-cse \
		-memref-expand \
		-arith-expand \
		-convert-vector-to-llvm \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-norm.out
	./next-norm.out || true

next-mhsa-qkv-aot-omp:
	@${MLIR_OPT} ./next-mhsa-qkv.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-matmul-parallel-vectorization-optimize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-convert-scf-to-openmp \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-mhsa.out
	./next-mhsa.out || true

next-mhsa-qkv-aot:
	@${MLIR_OPT} ./next-mhsa-qkv.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-matmul-parallel-vectorization-optimize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-mhsa.out
	./next-mhsa.out || true

next-mhsa-core-aot-omp:
	@${MLIR_OPT} ./next-mhsa-core.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-matmul-parallel-vectorization-optimize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-mhsa-core.out
	./next-mhsa-core.out || true

next-mhsa-core-aot:
	@${MLIR_OPT} ./next-mhsa-core.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-matmul-parallel-vectorization-optimize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-mhsa-core.out
	./next-mhsa-core.out || true

next-mhsa-context-aot-omp:
	@${MLIR_OPT} ./next-mhsa-context.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-matmul-parallel-vectorization-optimize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-mhsa-context.out
	./next-mhsa-context.out || true

next-mhsa-context-aot:
	@${MLIR_OPT} ./next-mhsa-context.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-matmul-parallel-vectorization-optimize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-mhsa-context.out
	./next-mhsa-context.out || true

next-ffn-no-reshape-aot-omp:
	@${MLIR_OPT} ./next-ffn-no-reshape.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" \
		-expand-strided-metadata \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-matmul-vectorization \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp=num-threads=16 \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-ffn-no-reshape.out
	./next-ffn-no-reshape.out || true

next-ffn-no-transpose-aot-omp:
	@${MLIR_OPT} ./next-ffn-no-transpose.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" \
		-expand-strided-metadata \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-matmul-vectorization \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp=num-threads=16 \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-ffn-no-transpose.out
	./next-ffn-no-transpose.out || true

next-reshape-lower:
	@${MLIR_OPT} ./next-reshape.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-convert-bufferization-to-memref \
		-matmul-vectorization \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp=num-threads=32 \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

next-reshape-aot-omp:
	@${MLIR_OPT} ./next-reshape.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-convert-bufferization-to-memref \
		-matmul-vectorization \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp=num-threads=32 \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-reshape.out
	./next-reshape.out || true

next-no-reshape-lower:
	@${MLIR_OPT} ./next-no-reshape.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-convert-bufferization-to-memref \
		-matmul-vectorization \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp=num-threads=32 \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

next-no-reshape-aot-omp:
	@${MLIR_OPT} ./next-no-reshape.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-convert-bufferization-to-memref \
		-matmul-vectorization \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp=num-threads=32 \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-no-reshape.out
	./next-no-reshape.out || true

next-ffn-aot-omp:
	@${MLIR_OPT} ./next-ffn.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-matmul-vectorization \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-mhsa-ffn.out
	./next-mhsa-ffn.out || true

next-ffn-aot:
	@${MLIR_OPT} ./next-ffn.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-matmul-parallel-vectorization-optimize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-mhsa-ffn.out
	./next-mhsa-ffn.out || true

next-output-run-omp:
	@${MLIR_OPT} ./next-output.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-matmul-parallel-vectorization-optimize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-output.out
	./next-output.out || true

next-output-run:
	@${MLIR_OPT} ./next-output.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${BUDDY_OPT} \
		-convert-elementwise-to-linalg \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-matmul-parallel-vectorization-optimize \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS} \
		-shared-libs=${OMP_LIB}

# ==============================================================================
# Evaluation of Flash Attention
# ==============================================================================

next-flash-attention-aot-omp:
	@${BUDDY_OPT} ./next-flash-attention.mlir \
		-lower-affine \
		-convert-scf-to-openmp \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-flash-attention.out
	./next-flash-attention.out || true

# ==============================================================================
# Evaluation of Other Kernels
# ==============================================================================

next-sgemm-parallel-unroll-vec-fixed-aot:
	@${BUDDY_OPT} ./next-sgemm.mlir \
		-matmul-vectorization \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
    ${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-sgemm-parallel-unroll-vec-fixed.out
	./next-sgemm-parallel-unroll-vec-fixed.out || true

next-sgemm-parallel-vec-fixed-aot:
	@${BUDDY_OPT} ./next-sgemm.mlir \
		-matmul-parallel-vectorization-optimize \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
    ${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-sgemm-parallel-vec-fixed.out
	./next-sgemm-parallel-vec-fixed.out || true

next-sgemm-unroll-vec-fixed-aot:
	@${BUDDY_OPT} ./next-sgemm-unroll-vec-fixed.mlir \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
    ${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-sgemm-unroll-vec-fixed.out
	./next-sgemm-unroll-vec-fixed.out || true

next-mmtb-aot:
	@${BUDDY_OPT} ./next-mmtb.mlir \
		-matmul-transpose-b-unroll-vec \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
    ${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-mmtb.out
	./next-mmtb.out || true

next-mmtb-sgemm-unroll-vec-aot:
	@${BUDDY_OPT} ./log.mlir \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
    ${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-mmtb-sgemm-unroll-vec.out
	./next-mmtb-sgemm-unroll-vec.out || true

next-batchmatmul-sgemm-unroll-vec-aot:
	@${BUDDY_OPT} ./next-batchmatmul-sgemm-unroll-vec.mlir \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-affine-parallelize \
		-lower-affine \
		-convert-scf-to-openmp \
		-func-bufferize-dynamic-offset \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-lower-affine \
		-cse \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
    ${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lomp -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-batchmatmul-sgemm-unroll-vec.out
	./next-batchmatmul-sgemm-unroll-vec.out || true

next-attention-lower:
	@${MLIR_OPT} ./next-attention.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts \
		-o ./log.mlir

next-attention-translate:
	@${MLIR_OPT} ./next-attention.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

next-attention-run:
	@${MLIR_OPT} ./next-attention.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		--one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-attention-loop-run:
	@${MLIR_OPT} ./next-attention-loop.mlir \
		-affine-loop-fusion \
		-lower-affine \
		--one-shot-bufferize="bufferize-function-boundaries" \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-attention-fusion-run:
	@${MLIR_OPT} ./next-attention-fusion.mlir \
		-affine-loop-fusion \
		-lower-affine \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-sigmoid-run:
	@${MLIR_OPT} ./next-sigmoid.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-eliminate-add-zero-run:
	@${MLIR_OPT} ./next-eliminate-add-zero.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-rope-run:
	@${MLIR_OPT} ./next-rope.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-lower-affine \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}


next-eliminate-identity-run:
	@${MLIR_OPT} ./next-eliminate-identity.mlir \
        -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
    ${MLIR_OPT} \
        -arith-expand \
        -eliminate-empty-tensors \
        -empty-tensor-to-alloc-tensor \
        -one-shot-bufferize="bufferize-function-boundaries" \
        -convert-linalg-to-affine-loops \
        -affine-loop-fusion \
        -lower-affine \
        -convert-vector-to-scf \
        -expand-strided-metadata \
        -convert-vector-to-llvm \
        -memref-expand \
        -arith-expand \
        -convert-arith-to-llvm \
        -finalize-memref-to-llvm \
        -convert-scf-to-cf \
		-convert-cf-to-llvm \
        -convert-openmp-to-llvm \
        -convert-arith-to-llvm \
        -convert-math-to-llvm \
        -convert-math-to-libm  \
        -convert-func-to-llvm \
        -reconcile-unrealized-casts | \
    ${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
        -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-sgemm-run:
	@${MLIR_OPT} ./next-sgemm.mlir \
		-convert-linalg-to-loops \
		-cse \
        -lower-affine \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-vector-to-llvm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
    ${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
        -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

next-transpose-lower:
	@${MLIR_OPT} ./next-transpose.mlir \
        -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
    ${MLIR_OPT} \
        -arith-expand \
        -eliminate-empty-tensors \
        -empty-tensor-to-alloc-tensor \
        -one-shot-bufferize="bufferize-function-boundaries" \
        -o log.mlir

next-transpose-run:
	@${MLIR_OPT} ./next-transpose.mlir \
        -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
    ${MLIR_OPT} \
        -arith-expand \
        -eliminate-empty-tensors \
        -empty-tensor-to-alloc-tensor \
        -one-shot-bufferize="bufferize-function-boundaries" \
        -convert-linalg-to-affine-loops \
        -affine-loop-fusion \
        -lower-affine \
        -convert-vector-to-scf \
        -expand-strided-metadata \
        -convert-vector-to-llvm \
        -memref-expand \
        -arith-expand \
        -convert-arith-to-llvm \
        -finalize-memref-to-llvm \
        -convert-scf-to-cf \
		-convert-cf-to-llvm \
        -convert-openmp-to-llvm \
        -convert-arith-to-llvm \
        -convert-math-to-llvm \
        -convert-math-to-libm  \
        -convert-func-to-llvm \
        -reconcile-unrealized-casts | \
    ${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
        -shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS}

next-transpose-vec-manual-run:
	@${MLIR_OPT} ./next-transpose-vec-manual.mlir \
        -convert-linalg-to-affine-loops \
        -affine-loop-fusion \
        -lower-affine \
		-convert-scf-to-openmp \
        -convert-vector-to-scf \
        -expand-strided-metadata \
        -convert-vector-to-llvm \
        -memref-expand \
        -arith-expand \
        -convert-arith-to-llvm \
        -finalize-memref-to-llvm \
        -convert-scf-to-cf \
		-convert-cf-to-llvm \
        -convert-openmp-to-llvm \
        -convert-arith-to-llvm \
        -convert-math-to-llvm \
        -convert-math-to-libm  \
        -convert-func-to-llvm \
        -reconcile-unrealized-casts | \
    ${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
        -shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS}

next-mask-run:
	@${MLIR_OPT} ./next-mask.mlir \
		-pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

pooling-nhwc-max-vec-run:
	@${BUDDY_OPT} ./pooling-nhwc-max-vec.mlir \
		-convert-linalg-to-loops \
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

next-compass-lower:
	@${MLIR_OPT} ./next-compass.mlir \
		-arith-expand \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-convert-linalg-to-affine-loops \
		-affine-loop-fusion \
		-lower-affine \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-memref-expand \
		-arith-expand \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-openmp-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts  \
		-o log.mlir

next-compass-run:
	@${MLIR_OPT} ./next-compass.mlir \
    -arith-expand \
    -eliminate-empty-tensors \
    -empty-tensor-to-alloc-tensor \
    -one-shot-bufferize="bufferize-function-boundaries" \
    -convert-linalg-to-affine-loops \
    -affine-loop-fusion \
    -lower-affine \
    -convert-vector-to-scf \
    -expand-strided-metadata \
    -convert-vector-to-llvm \
    -memref-expand \
    -arith-expand \
    -convert-arith-to-llvm \
    -finalize-memref-to-llvm \
    -convert-scf-to-cf \
    -convert-openmp-to-llvm \
    -convert-arith-to-llvm \
    -convert-math-to-llvm \
    -convert-math-to-libm  \
    -convert-func-to-llvm \
    -reconcile-unrealized-casts  | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} \
		-shared-libs=${MLIR_C_RUNNER_UTILS}

tosa-matmul-transpose2-lower:
	@${BUDDY_OPT} ./tosa-matmultranspose2.mlir \
			-transpose-fusion-vectorization \
			-o log.mlir

tosa-matmul-transpose2-run:
	@${BUDDY_OPT} ./tosa-matmultranspose2.mlir \
			-pass-pipeline "builtin.module(transpose-fusion-vectorization, func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" | \
    ${BUDDY_OPT} \
		-eliminate-empty-tensors \
		-convert-tensor-to-linalg \
		-linalg-bufferize \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-func-bufferize \
		-arith-bufferize \
		-tensor-bufferize \
		-buffer-deallocation \
		-finalizing-bufferize \
		-convert-vector-to-scf \
		-expand-strided-metadata \
		-convert-vector-to-llvm \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-scf-to-cf \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

tosa-matmul-transpose2-vec-run:
	@${BUDDY_OPT} ./tosa-matmultranspose2-vec.mlir\
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
		-convert-openmp-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm  \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
