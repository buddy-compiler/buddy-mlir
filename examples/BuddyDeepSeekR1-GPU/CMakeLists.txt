set(BUFFERIZE_SIMPLE_OPTS "bufferize-function-boundaries")
set(TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")

set(CUDA_SM "sm_80" CACHE STRING "CUDA SM target (e.g. sm_80, sm_86, sm_89)")
set(CUDA_FEATURES "+ptx71" CACHE STRING "CUDA PTX features (e.g. +ptx71)")
set(NVVM_PIPELINE_OPTS "cubin-chip=${CUDA_SM} cubin-features=${CUDA_FEATURES} cubin-format=fatbin")

# ---------------------------------------------------------------------------
# Import: generate MLIR files and parameters using the DeepSeekR1 import script
# ---------------------------------------------------------------------------
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1-gpu.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating DeepSeekR1 MLIR and parameter files for GPU..."
)

# ---------------------------------------------------------------------------
# Forward prefill: standard LLVM pipeline (no OpenMP, no CPU vectorization)
# The forward orchestrates calls to the GPU subgraph; unified memory makes all
# allocations accessible from both CPU and GPU without explicit transfers.
# ---------------------------------------------------------------------------
add_custom_command(
  OUTPUT forward_prefill.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt
            ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.mlir
            -simplify-tosa-reshape |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -canonicalize
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -cse
            -canonicalize
            -optimize-allocation-liveness
            -convert-linalg-to-affine-loops
            -convert-vector-to-scf
            -lower-affine
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.mlir
  COMMENT "Building forward_prefill.o (GPU)"
  VERBATIM)

# ---------------------------------------------------------------------------
# Subgraph prefill: GPU pipeline
# linalg → parallel-loops → gpu-map → gpu-kernel-outlining → NVVM
# ---------------------------------------------------------------------------
add_custom_command(
  OUTPUT subgraph_prefill.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill.mlir
            -simplify-tosa-reshape |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -canonicalize
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -cse
            -canonicalize
            -optimize-allocation-liveness
            -convert-strided-copy-to-parallel
            -matmul-gpu-tiling
            -convert-linalg-to-parallel-loops
            -canonicalize
            -nest-parallel-loops-for-gpu
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -legalize-shmem-outlining
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -llvm-request-c-wrappers
            -gpu-lower-to-nvvm-pipeline=${NVVM_PIPELINE_OPTS} |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill.mlir
  COMMENT "Building subgraph_prefill.o (GPU)"
  VERBATIM)

# ---------------------------------------------------------------------------
# Forward decode: same pipeline as forward prefill
# ---------------------------------------------------------------------------
add_custom_command(
  OUTPUT forward_decode.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt
            ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.mlir
            -simplify-tosa-reshape |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -canonicalize
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -cse
            -canonicalize
            -optimize-allocation-liveness
            -convert-memcpy-to-gpu=process-args=0
            -convert-linalg-to-affine-loops
            -convert-vector-to-scf
            -lower-affine
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -llvm-request-c-wrappers
            -gpu-to-llvm |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.mlir
  COMMENT "Building forward_decode.o (GPU)"
  VERBATIM)

# ---------------------------------------------------------------------------
# Subgraph decode: GPU pipeline (with matmul-scalar simplification for decode)
# ---------------------------------------------------------------------------
add_custom_command(
  OUTPUT subgraph_decode.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode.mlir
            -simplify-tosa-reshape
            -simplify-tosa-matmul-scalar |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -canonicalize
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -cse
            -canonicalize
            -optimize-allocation-liveness
            -convert-strided-copy-to-parallel
            -matmul-gpu-tiling
            -convert-linalg-to-parallel-loops
            -canonicalize
            -nest-parallel-loops-for-gpu
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -legalize-shmem-outlining
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -convert-memcpy-to-gpu=process-args=0
            -gpu-async-region
            -canonicalize |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -llvm-request-c-wrappers
            -gpu-lower-to-nvvm-pipeline=${NVVM_PIPELINE_OPTS} |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode.mlir
  COMMENT "Building subgraph_decode.o (GPU)"
  VERBATIM)

# ---------------------------------------------------------------------------
# Static library
# ---------------------------------------------------------------------------
add_library(DEEPSEEKR1_GPU STATIC
  forward_prefill.o
  subgraph_prefill.o
  forward_decode.o
  subgraph_decode.o)

SET_SOURCE_FILES_PROPERTIES(
  forward_prefill.o subgraph_prefill.o forward_decode.o subgraph_decode.o
  PROPERTIES EXTERNAL_OBJECT true GENERATED true)

SET_TARGET_PROPERTIES(DEEPSEEKR1_GPU PROPERTIES LINKER_LANGUAGE C)

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(buddy-deepseek-r1-gpu-run buddy-deepseek-r1-gpu-main.cpp)

# DEEPSEEKR1_GPU_EXAMPLE_PATH points to the CPU source directory (contains vocab.txt).
# DEEPSEEKR1_GPU_EXAMPLE_BUILD_PATH points to this GPU build directory (contains arg0.data).
set(DEEPSEEKR1_GPU_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../BuddyDeepSeekR1/)
set(DEEPSEEKR1_GPU_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(buddy-deepseek-r1-gpu-run PRIVATE
  DEEPSEEKR1_GPU_EXAMPLE_PATH="${DEEPSEEKR1_GPU_EXAMPLE_PATH}"
  DEEPSEEKR1_GPU_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_GPU_EXAMPLE_BUILD_PATH}/"
)

target_include_directories(buddy-deepseek-r1-gpu-run PRIVATE /usr/local/cuda/include)

target_link_directories(buddy-deepseek-r1-gpu-run PRIVATE ${LLVM_LIBRARY_DIR} /usr/local/cuda/lib64)

set(BUDDY_DEEPSEEKR1_GPU_LIBS
  DEEPSEEKR1_GPU
  mlir_cuda_runtime
  mlir_c_runner_utils
  mlir_runner_utils
  cudart
)

target_link_libraries(buddy-deepseek-r1-gpu-run ${BUDDY_DEEPSEEKR1_GPU_LIBS})
