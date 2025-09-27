# Frontend: Generate MLIR files from PyTorch model
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-transformer.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
)


# Forward: One-step compilation to object file
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize="bufferize-function-boundaries"
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

# Subgraph: One-step compilation to object file
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp
            -func-bufferize-dynamic-offset
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
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
    COMMENT "Building subgraph.o"
    VERBATIM)

# Executable: Link object file with runner
add_executable(transformer-runner
    ${CMAKE_CURRENT_SOURCE_DIR}/transformer_runner.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph.o
)

# Set properties for the executable
set_target_properties(transformer-runner PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Include Buddy headers
target_include_directories(transformer-runner PRIVATE
    ${CMAKE_SOURCE_DIR}/frontend/Interfaces
)

# Define paths for the executable
target_compile_definitions(transformer-runner PRIVATE
    BUDDY_TRANSFORMER_EXAMPLE_PATH="${CMAKE_CURRENT_SOURCE_DIR}"
    BUDDY_TRANSFORMER_EXAMPLE_BUILD_PATH="${CMAKE_CURRENT_BINARY_DIR}"
)

# Set up library directories and linking
target_link_directories(transformer-runner PRIVATE ${LLVM_LIBRARY_DIR})

# Define libraries to link
set(BUDDY_TRANSFORMER_LIBS
    mlir_c_runner_utils
    omp
)

# Link with required libraries
target_link_libraries(transformer-runner ${BUDDY_TRANSFORMER_LIBS})

# Create targets for building the MLIR files
add_custom_target(buddy-transformer-frontend
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
            ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
    COMMENT "Building frontend MLIR files and Buddy Graph logs"
)

add_custom_target(buddy-transformer-codegen
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward.o
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph.o
    COMMENT "Generating object files"
)

add_custom_target(buddy-transformer-executable
    DEPENDS transformer-runner
    COMMENT "Building transformer block performance test executable"
)

add_custom_target(buddy-transformer
    DEPENDS buddy-transformer-executable
    COMMENT "Building complete transformer block pipeline with executable"
)

# Add dependency to ensure object file is built first
add_dependencies(transformer-runner buddy-transformer-codegen)

# Clean target to remove generated files
add_custom_target(buddy-transformer-clean
    COMMAND ${CMAKE_COMMAND} -E remove -f
        ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
        ${CMAKE_CURRENT_BINARY_DIR}/forward.o
        ${CMAKE_CURRENT_BINARY_DIR}/subgraph.o
        ${CMAKE_CURRENT_BINARY_DIR}/graph.log
        ${CMAKE_CURRENT_BINARY_DIR}/graph_fused.log
    COMMAND ${CMAKE_COMMAND} -E remove -f
        ${CMAKE_CURRENT_BINARY_DIR}/transformer-runner
    COMMENT "Cleaning BuddyTransformer generated files"
)
