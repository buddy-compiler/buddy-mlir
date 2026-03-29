BUDDY_BUILD_DIR := ../../build
LLVM_BUILD_DIR  := ../../llvm/build

BUDDY_OPT    := ${BUDDY_BUILD_DIR}/bin/buddy-opt
MLIR_OPT     := ${LLVM_BUILD_DIR}/bin/mlir-opt
MLIR_RUNNER  := ${LLVM_BUILD_DIR}/bin/mlir-runner
MLIR_TRANSLATE := ${LLVM_BUILD_DIR}/bin/mlir-translate
CLANG := ${LLVM_BUILD_DIR}/bin/clang
MLIR_LIB := ${LLVM_BUILD_DIR}/lib/

UNAME_M := $(shell uname -m)
ifeq ($(findstring riscv,$(UNAME_M)),riscv)
MARCH_FLAG := -march=rv64gcv
else
MARCH_FLAG := -march=native
endif

MLIR_RUNNER_UTILS   := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.so
MLIR_CUDA_RUNTIME   := ${LLVM_BUILD_DIR}/lib/libmlir_cuda_runtime.so

CUDA_SM       ?= sm_80
CUDA_FEATURES ?= +ptx71
NVVM_PIPELINE  := cubin-chip=${CUDA_SM} cubin-features=${CUDA_FEATURES} cubin-format=fatbin
TOSA_PIPELINE  := builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))
# ==============================================================================
# LLM Operator GPU Evaluation
# ==============================================================================

next-qwen3-prefill-run:
	@${BUDDY_OPT} ./next-qwen3-prefill.mlir \
		-simplify-tosa-reshape | \
	${BUDDY_OPT} \
		-pass-pipeline "${TOSA_PIPELINE}" | \
	${BUDDY_OPT} \
		"-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
		-convert-linalg-to-parallel-loops \
		-canonicalize \
		-gpu-map-parallel-loops \
		-convert-parallel-loops-to-gpu \
		-gpu-kernel-outlining \
		-buffer-hoisting \
		-canonicalize -cse | \
	${BUDDY_OPT} \
		-convert-memcpy-to-gpu \
		-gpu-async-region \
		-canonicalize | \
	${MLIR_OPT} \
		"-gpu-lower-to-nvvm-pipeline=${NVVM_PIPELINE}" | \
	${MLIR_OPT} \
		-gpu-to-llvm \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_RUNNER} \
		-entry-point-result=void \
		-shared-libs=${MLIR_CUDA_RUNTIME},${MLIR_C_RUNNER_UTILS},${MLIR_RUNNER_UTILS}


next-qwen3-decode-run:
	@${BUDDY_OPT} ./next-qwen3-decode.mlir \
		-simplify-tosa-reshape | \
	${BUDDY_OPT} \
		-pass-pipeline "${TOSA_PIPELINE}" | \
	${BUDDY_OPT} \
		"-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
		-convert-linalg-to-parallel-loops \
		-canonicalize \
		-gpu-map-parallel-loops \
		-convert-parallel-loops-to-gpu \
		-gpu-kernel-outlining \
		-buffer-hoisting \
		-canonicalize -cse | \
	${BUDDY_OPT} \
		-convert-memcpy-to-gpu \
		-gpu-async-region \
		-canonicalize | \
	${MLIR_OPT} \
		"-gpu-lower-to-nvvm-pipeline=${NVVM_PIPELINE}" | \
	${MLIR_OPT} \
		-gpu-to-llvm \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_RUNNER} \
		-entry-point-result=void \
		-shared-libs=${MLIR_CUDA_RUNTIME},${MLIR_C_RUNNER_UTILS},${MLIR_RUNNER_UTILS}


next-kvcache-run:
	@${BUDDY_OPT} ./next-kvcache.mlir | \
	${BUDDY_OPT} \
		"-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
		-convert-linalg-to-parallel-loops \
		-canonicalize \
		-gpu-map-parallel-loops \
		-convert-parallel-loops-to-gpu \
		-gpu-kernel-outlining \
		-buffer-hoisting \
		-canonicalize -cse | \
	${BUDDY_OPT} \
		-convert-memcpy-to-gpu \
		-gpu-async-region \
		-canonicalize | \
	${MLIR_OPT} \
		"-gpu-lower-to-nvvm-pipeline=${NVVM_PIPELINE}" | \
	${MLIR_OPT} \
		-gpu-to-llvm \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_RUNNER} \
		-entry-point-result=void \
		-shared-libs=${MLIR_CUDA_RUNTIME},${MLIR_C_RUNNER_UTILS},${MLIR_RUNNER_UTILS}


next-rmsnorm-run:
	@${BUDDY_OPT} ./next-rmsnorm.mlir \
		-simplify-tosa-reshape | \
	${BUDDY_OPT} \
		-pass-pipeline "${TOSA_PIPELINE}" | \
	${BUDDY_OPT} \
		"-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
		-convert-linalg-to-parallel-loops \
		-canonicalize \
		-gpu-map-parallel-loops \
		-convert-parallel-loops-to-gpu \
		-gpu-kernel-outlining \
		-buffer-hoisting \
		-canonicalize -cse | \
	${BUDDY_OPT} \
		-convert-memcpy-to-gpu \
		-gpu-async-region \
		-canonicalize | \
	${MLIR_OPT} \
		"-gpu-lower-to-nvvm-pipeline=${NVVM_PIPELINE}" | \
	${MLIR_OPT} \
		-gpu-to-llvm \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_RUNNER} \
		-entry-point-result=void \
		-shared-libs=${MLIR_CUDA_RUNTIME},${MLIR_C_RUNNER_UTILS},${MLIR_RUNNER_UTILS}


next-rope-run:
	@${BUDDY_OPT} ./next-rope.mlir \
		-simplify-tosa-reshape | \
	${BUDDY_OPT} \
		-pass-pipeline "${TOSA_PIPELINE}" | \
	${BUDDY_OPT} \
		"-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
		-convert-linalg-to-parallel-loops \
		-canonicalize \
		-gpu-map-parallel-loops \
		-convert-parallel-loops-to-gpu \
		-gpu-kernel-outlining \
		-buffer-hoisting \
		-canonicalize -cse | \
	${BUDDY_OPT} \
		-convert-memcpy-to-gpu \
		-gpu-async-region \
		-canonicalize | \
	${MLIR_OPT} \
		"-gpu-lower-to-nvvm-pipeline=${NVVM_PIPELINE}" | \
	${MLIR_OPT} \
		-gpu-to-llvm \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_RUNNER} \
		-entry-point-result=void \
		-shared-libs=${MLIR_CUDA_RUNTIME},${MLIR_C_RUNNER_UTILS},${MLIR_RUNNER_UTILS}


next-embedding-run:
	@${BUDDY_OPT} ./next-embedding.mlir \
		-simplify-tosa-reshape | \
	${BUDDY_OPT} \
		-pass-pipeline "${TOSA_PIPELINE}" | \
	${BUDDY_OPT} \
		"-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
		-convert-linalg-to-parallel-loops \
		-canonicalize \
		-gpu-map-parallel-loops \
		-convert-parallel-loops-to-gpu \
		-gpu-kernel-outlining \
		-buffer-hoisting \
		-canonicalize -cse | \
	${BUDDY_OPT} \
		-convert-memcpy-to-gpu \
		-gpu-async-region \
		-canonicalize | \
	${MLIR_OPT} \
		"-gpu-lower-to-nvvm-pipeline=${NVVM_PIPELINE}" | \
	${MLIR_OPT} \
		-gpu-to-llvm \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_RUNNER} \
		-entry-point-result=void \
		-shared-libs=${MLIR_CUDA_RUNTIME},${MLIR_C_RUNNER_UTILS},${MLIR_RUNNER_UTILS}


next-ffn-run:
	@${BUDDY_OPT} ./next-ffn.mlir \
		-simplify-tosa-reshape | \
	${BUDDY_OPT} \
		-pass-pipeline "${TOSA_PIPELINE}" | \
	${BUDDY_OPT} \
		"-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
		-convert-linalg-to-parallel-loops \
		-canonicalize \
		-gpu-map-parallel-loops \
		-convert-parallel-loops-to-gpu \
		-gpu-kernel-outlining \
		-buffer-hoisting \
		-canonicalize -cse | \
	${BUDDY_OPT} \
		-convert-memcpy-to-gpu \
		-gpu-async-region \
		-canonicalize | \
	${MLIR_OPT} \
		"-gpu-lower-to-nvvm-pipeline=${NVVM_PIPELINE}" | \
	${MLIR_OPT} \
		-gpu-to-llvm \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_RUNNER} \
		-entry-point-result=void \
		-shared-libs=${MLIR_CUDA_RUNTIME},${MLIR_C_RUNNER_UTILS},${MLIR_RUNNER_UTILS}


next-attention-run:
	@${BUDDY_OPT} ./next-attention.mlir \
		-simplify-tosa-reshape | \
	${BUDDY_OPT} \
		-pass-pipeline "${TOSA_PIPELINE}" | \
	${BUDDY_OPT} \
		"-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-convert-linalg-to-parallel-loops \
		-canonicalize \
		-gpu-map-parallel-loops \
		-convert-parallel-loops-to-gpu \
		-gpu-kernel-outlining \
		-buffer-hoisting \
		-canonicalize -cse | \
	${BUDDY_OPT} \
		-convert-memcpy-to-gpu \
		-gpu-async-region \
		-canonicalize | \
	${MLIR_OPT} \
		"-gpu-lower-to-nvvm-pipeline=${NVVM_PIPELINE}" | \
	${MLIR_OPT} \
		-gpu-to-llvm \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-finalize-memref-to-llvm \
		-convert-arith-to-llvm \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_RUNNER} \
		-entry-point-result=void \
		-shared-libs=${MLIR_CUDA_RUNTIME},${MLIR_C_RUNNER_UTILS},${MLIR_RUNNER_UTILS}


# ==============================================================================
# LLM Operator CPU Evaluation
# ==============================================================================


next-attention-cpu-run:
	@${BUDDY_OPT} ./next-attention.mlir \
		-pass-pipeline "${TOSA_PIPELINE}" | \
	${BUDDY_OPT} \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-convert-elementwise-to-linalg \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-expand-strided-metadata \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-convert-linalg-to-loops \
		-convert-scf-to-cf \
		-convert-bufferization-to-memref \
		-cse \
		-canonicalize \
		-memref-expand \
		-arith-expand \
		-convert-vector-to-llvm \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-attention-cpu-run.out
	./next-attention-cpu-run.out || true


next-qwen3-prefill-cpu-run:
	@${BUDDY_OPT} ./next-qwen3-prefill.mlir \
		-pass-pipeline "${TOSA_PIPELINE}" | \
	${BUDDY_OPT} \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-convert-elementwise-to-linalg \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-expand-strided-metadata \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-bufferization-to-memref \
		-cse \
		-canonicalize \
		-canonicalize \
		-memref-expand \
		-arith-expand \
		-convert-vector-to-llvm \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-qwen3-prefill-cpu-run.out
	./next-qwen3-prefill-cpu-run.out || true


next-qwen3-decode-cpu-run:
	@${BUDDY_OPT} ./next-qwen3-decode.mlir \
		-pass-pipeline "${TOSA_PIPELINE}" | \
	${BUDDY_OPT} \
		-eliminate-empty-tensors \
		-empty-tensor-to-alloc-tensor \
		-convert-elementwise-to-linalg \
		-one-shot-bufferize="bufferize-function-boundaries" \
		-expand-strided-metadata \
		-ownership-based-buffer-deallocation \
		-buffer-deallocation-simplification \
		-bufferization-lower-deallocations \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-bufferization-to-memref \
		-cse \
		-canonicalize \
		-memref-expand \
		-arith-expand \
		-convert-vector-to-llvm \
		-convert-arith-to-llvm \
		-finalize-memref-to-llvm \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} -mlir-to-llvmir | \
	${CLANG} -x ir - \
		${MARCH_FLAG} -O3 \
		-L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -lm \
		-Wl,-rpath,${MLIR_LIB} \
		-o next-qwen3-decode-cpu-run.out
	./next-qwen3-decode-cpu-run.out || true
