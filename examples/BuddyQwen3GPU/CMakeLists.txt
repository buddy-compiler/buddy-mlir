set(TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")

set(CUDA_SM "sm_80" CACHE STRING "CUDA SM target (e.g. sm_80, sm_86)")
set(CUDA_FEATURES "+ptx71" CACHE STRING "CUDA PTX features (e.g. +ptx71)")
set(NVVM_PIPELINE_OPTS "cubin-chip=${CUDA_SM} cubin-features=${CUDA_FEATURES} cubin-format=fatbin")

set(BUFFERIZE_OPTS "bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map")

# ---------------------------------------------------------------------------
# Prefill kernel: next-qwen3-prefill-kernel.mlir -> prefill_kernel.o
# ---------------------------------------------------------------------------
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/prefill_kernel.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt
            ${CMAKE_CURRENT_SOURCE_DIR}/next-qwen3-prefill-kernel.mlir
            -simplify-tosa-reshape |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -one-shot-bufferize=${BUFFERIZE_OPTS}
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -convert-memcpy-to-gpu
            -gpu-async-region
            -canonicalize
            -llvm-request-c-wrappers |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -gpu-lower-to-nvvm-pipeline=${NVVM_PIPELINE_OPTS} |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -gpu-to-llvm
            -convert-vector-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -finalize-memref-to-llvm
            -convert-arith-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/prefill_kernel.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/next-qwen3-prefill-kernel.mlir
  COMMENT "Building prefill_kernel.o"
  VERBATIM)

# ---------------------------------------------------------------------------
# Decode kernel: next-qwen3-decode-kernel.mlir -> decode_kernel.o
# ---------------------------------------------------------------------------
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/decode_kernel.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt
            ${CMAKE_CURRENT_SOURCE_DIR}/next-qwen3-decode-kernel.mlir
            -simplify-tosa-reshape |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -one-shot-bufferize=${BUFFERIZE_OPTS}
            -convert-linalg-to-parallel-loops
            -canonicalize
            -gpu-map-parallel-loops
            -convert-parallel-loops-to-gpu
            -gpu-kernel-outlining
            -buffer-hoisting
            -canonicalize
            -cse |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -convert-memcpy-to-gpu
            -gpu-async-region
            -canonicalize
            -llvm-request-c-wrappers |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -gpu-lower-to-nvvm-pipeline=${NVVM_PIPELINE_OPTS} |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -gpu-to-llvm
            -convert-vector-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -finalize-memref-to-llvm
            -convert-arith-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/decode_kernel.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/next-qwen3-decode-kernel.mlir
  COMMENT "Building decode_kernel.o"
  VERBATIM)

# ---------------------------------------------------------------------------
# Static library bundling both kernel objects
# ---------------------------------------------------------------------------
add_library(QWEN3_NEXTGPU STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/prefill_kernel.o
  ${CMAKE_CURRENT_BINARY_DIR}/decode_kernel.o)

set_source_files_properties(
  ${CMAKE_CURRENT_BINARY_DIR}/prefill_kernel.o
  ${CMAKE_CURRENT_BINARY_DIR}/decode_kernel.o
  PROPERTIES EXTERNAL_OBJECT true GENERATED true)

set_target_properties(QWEN3_NEXTGPU PROPERTIES LINKER_LANGUAGE CXX)

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(buddy-qwen3-nextgpu-run
  ${CMAKE_CURRENT_SOURCE_DIR}/buddy-qwen3-nextgpu-main.cpp)

target_link_directories(buddy-qwen3-nextgpu-run PRIVATE ${LLVM_LIBRARY_DIR})

target_link_libraries(buddy-qwen3-nextgpu-run
  QWEN3_NEXTGPU
  mlir_cuda_runtime
  mlir_c_runner_utils
  mlir_runner_utils)

target_compile_definitions(buddy-qwen3-nextgpu-run PRIVATE
  QWEN3_NEXTGPU_BUILD_PATH="${CMAKE_CURRENT_BINARY_DIR}/")

# Copy vocab.txt to build directory for convenience
add_custom_command(TARGET buddy-qwen3-nextgpu-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/../BuddyQwen3/vocab.txt
    ${CMAKE_CURRENT_BINARY_DIR}/vocab.txt
  COMMENT "Copying vocab.txt to build directory")
