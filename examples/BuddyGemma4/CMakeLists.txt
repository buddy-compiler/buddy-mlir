# Define bufferize options as variables to avoid CMake quote parsing issues
set(BUFFERIZE_FULL_OPTS "unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries")
set(BUFFERIZE_SIMPLE_OPTS "bufferize-function-boundaries")
set(TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")

# Detect RISC-V Vector (RVV) platform
if(HAVE_LOCAL_RVV)
  set(IS_RVV_PLATFORM ON)
  message(STATUS "RISC-V Vector (RVV) platform detected")
endif()

# Set platform-specific variables
if(IS_RVV_PLATFORM)
  set(OPENMP_NUM_THREADS "num-threads=32")
  set(LLC_RISCV_ATTRS "-mattr=+m,+d,+v")
  set(RISCV_COMPILE_OPTS "-march=rv64gcv;-mabi=lp64d;-O3;-Wall")
  set(RISCV_LINK_OPTS "-march=rv64gcv;-mabi=lp64d")
else()
  set(OPENMP_NUM_THREADS "num-threads=48")
  set(LLC_RISCV_ATTRS "")
endif()

# Skip import commands on RVV platform
if(NOT IS_RVV_PLATFORM)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_e2b.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill_e2b.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_e2b.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_e2b.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/arg0_e2b.data
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-gemma4.py
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}
            --precision f32
    COMMENT "Generating Gemma4-E2B MLIR and data files..."
  )
endif()

add_custom_command(
  OUTPUT forward_prefill_e2b.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_e2b.mlir
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
            -matmul-vectorization-blis
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
        ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
        ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_e2b.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_e2b.mlir
  COMMENT "Building forward_prefill_e2b.o"
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_prefill_e2b.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill_e2b.mlir
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
            -matmul-vectorization-blis
            -batchmatmul-optimize
            -batchmatmul-transpose-b-vectorization
            -convert-linalg-to-affine-loops
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=num-threads=48
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill_e2b.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill_e2b.mlir
    COMMENT "Building subgraph_prefill_e2b.o"
    VERBATIM)

add_custom_command(
  OUTPUT forward_decode_e2b.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_e2b.mlir
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
            -matmul-vectorization-blis
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
        ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
        ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_e2b.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_e2b.mlir
  COMMENT "Building forward_decode_e2b.o"
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_decode_e2b.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_e2b.mlir
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
            -eliminate-memref-copy
            -assume-tight-memref-layout
            -staticize-memref-layout
            -matmul-vectorization-decode=vector-size=128
            -batch-matmul-vectorization-decode=vector-size=128
            -batchmatmul-transpose-b-vectorization=vector-size=16
            -convert-linalg-to-affine-loops
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode_e2b.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_e2b.mlir
    COMMENT "Building subgraph_decode_e2b.o"
    VERBATIM)


add_library(GEMMA4_E2B STATIC forward_prefill_e2b.o subgraph_prefill_e2b.o forward_decode_e2b.o subgraph_decode_e2b.o)

SET_SOURCE_FILES_PROPERTIES(
  template.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true)

SET_TARGET_PROPERTIES(
  GEMMA4_E2B
  PROPERTIES
  LINKER_LANGUAGE C)


add_executable(buddy-gemma4-e2b-run buddy-gemma4-e2b-main.cpp)

set(GEMMA4_E2B_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(GEMMA4_E2B_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(buddy-gemma4-e2b-run PRIVATE
  GEMMA4_E2B_EXAMPLE_PATH="${GEMMA4_E2B_EXAMPLE_PATH}/"
  GEMMA4_E2B_EXAMPLE_BUILD_PATH="${GEMMA4_E2B_EXAMPLE_BUILD_PATH}/"
)


target_link_directories(buddy-gemma4-e2b-run PRIVATE ${LLVM_LIBRARY_DIR})

set(BUDDY_GEMMA4_E2B_LIBS
  GEMMA4_E2B
  mlir_c_runner_utils
  omp
)

if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_GEMMA4_E2B_LIBS mimalloc)
endif()

target_link_libraries(buddy-gemma4-e2b-run ${BUDDY_GEMMA4_E2B_LIBS})
