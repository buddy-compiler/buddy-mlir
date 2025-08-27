# Frontend: Generate MLIR files from PyTorch model
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-attention.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
)


# Midend: Optimize forward.mlir (Stage 1 - Midend optimization)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward-midend.mlir
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize="bufferize-function-boundaries"
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -lower-affine
            -affine-loop-fusion
            -affine-parallelize
            -convert-scf-to-openmp
            -convert-vector-to-scf
            -expand-strided-metadata
            -convert-scf-to-cf
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward-midend.mlir
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  COMMENT "Optimizing forward.mlir (midend passes)"
  VERBATIM)

# Backend: Optimize forward.mlir (Stage 2 - Backend optimization)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward-optimized.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward-midend.mlir
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward-optimized.mlir
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward-midend.mlir
  COMMENT "Optimizing forward.mlir (backend passes)"
  VERBATIM)

# Midend: Optimize subgraph0.mlir (Stage 1 - Midend optimization)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-midend.mlir
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -convert-elementwise-to-linalg
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize="bufferize-function-boundaries"
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -lower-affine
            -affine-loop-fusion
            -affine-parallelize
            -convert-scf-to-openmp
            -func-bufferize-dynamic-offset
            -convert-vector-to-scf
            -expand-strided-metadata
            -cse
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-midend.mlir
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
    COMMENT "Optimizing subgraph0.mlir (midend passes)"
    VERBATIM)

# Backend: Optimize subgraph0.mlir (Stage 2 - Backend optimization)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-optimized.mlir
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-midend.mlir
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
            -reconcile-unrealized-casts
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-optimized.mlir
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-midend.mlir
    COMMENT "Optimizing subgraph0.mlir (backend passes)"
    VERBATIM)

# Code Generation: Generate LLVM IR and object files
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-translate --mlir-to-llvmir
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-optimized.mlir
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-optimized.mlir
    COMMENT "Generating LLVM IR from optimized subgraph0.mlir"
    VERBATIM)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
    COMMENT "Generating object file from LLVM IR"
    VERBATIM)

# Executable: Link object file with runner
add_executable(attention-runner
    ${CMAKE_CURRENT_SOURCE_DIR}/attention_runner.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
)

# Set properties for the executable
set_target_properties(attention-runner PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Include Buddy headers
target_include_directories(attention-runner PRIVATE
    ${CMAKE_SOURCE_DIR}/frontend/Interfaces
)

# Set up library directories and linking
target_link_directories(attention-runner PRIVATE ${LLVM_LIBRARY_DIR})

# Define libraries to link
set(BUDDY_ATTENTION_LIBS
    mlir_c_runner_utils
    omp
)

# Link with required libraries
target_link_libraries(attention-runner ${BUDDY_ATTENTION_LIBS})

# Add dependency to ensure object file is built first
add_dependencies(attention-runner buddy-attention-codegen)


# Create targets for building the MLIR files
add_custom_target(buddy-attention-frontend
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
            ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
    COMMENT "Building frontend MLIR files and Buddy Graph logs"
)

add_custom_target(buddy-attention-midend
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward-midend.mlir
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-midend.mlir
    COMMENT "Building midend optimized attention MLIR files"
)

add_custom_target(buddy-attention-backend
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward-optimized.mlir
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-optimized.mlir
    COMMENT "Building backend optimized attention MLIR files"
)

add_custom_target(buddy-attention-codegen
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
    COMMENT "Generating LLVM IR and object files"
)

add_custom_target(buddy-attention-executable
    DEPENDS attention-runner
    COMMENT "Building attention performance test executable"
)

add_custom_target(buddy-attention
    DEPENDS buddy-attention-executable
    COMMENT "Building complete attention pipeline with executable"
)


# Clean target to remove generated files
add_custom_target(buddy-attention-clean
    COMMAND ${CMAKE_COMMAND} -E remove -f
        ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
        ${CMAKE_CURRENT_BINARY_DIR}/forward-midend.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-midend.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/forward-optimized.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-optimized.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
        ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
        ${CMAKE_CURRENT_BINARY_DIR}/graph.log
        ${CMAKE_CURRENT_BINARY_DIR}/graph_fused.log
    COMMAND ${CMAKE_COMMAND} -E remove -f
        ${CMAKE_CURRENT_BINARY_DIR}/attention-runner
    COMMENT "Cleaning BuddyAttention generated files"
)
