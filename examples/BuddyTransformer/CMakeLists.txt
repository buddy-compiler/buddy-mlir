# Frontend: Generate MLIR files from PyTorch model
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-transformer.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
)

# One-step compilation

# Forward: One-step compilation to object file
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward-onestep.o
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward-onestep.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  COMMENT "Building forward-onestep.o (one-step compilation)"
  VERBATIM)

# Subgraph: One-step compilation to object file
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph-onestep.o
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
            # -func-bufferize-dynamic-offset
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
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph-onestep.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
    COMMENT "Building subgraph-onestep.o (one-step compilation)"
    VERBATIM)

# Subgraph Timed: One-step compilation with timing instrumentation
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph-timed.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0_timed.mlir
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
            # -func-bufferize-dynamic-offset
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
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph-timed.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0_timed.mlir
    COMMENT "Building subgraph-timed.o (with timing instrumentation)"
    VERBATIM)

# Staged compilation (for analysis)

# Forward: Midend optimization
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward-midend.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward-midend.mlir
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  COMMENT "Building forward-midend.mlir (midend optimization)"
  VERBATIM)

# Forward: Backend lowering
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward-backend.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward-midend.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward-backend.mlir
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward-midend.mlir
  COMMENT "Building forward-backend.mlir (backend lowering)"
  VERBATIM)

# Forward: Code generation
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.ll
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-translate --mlir-to-llvmir
          ${CMAKE_CURRENT_BINARY_DIR}/forward-backend.mlir
        -o ${CMAKE_CURRENT_BINARY_DIR}/forward.ll
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward-backend.mlir
  COMMENT "Building forward.ll (LLVM IR)"
  VERBATIM)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
          ${CMAKE_CURRENT_BINARY_DIR}/forward.ll
        -o ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward.ll
  COMMENT "Building forward.o (object file)"
  VERBATIM)

# Subgraph: Midend optimization
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-midend.mlir
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
            # -func-bufferize-dynamic-offset
            -cse
            -memref-expand
            -arith-expand
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-midend.mlir
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
    COMMENT "Building subgraph0-midend.mlir (midend optimization)"
    VERBATIM)

# Subgraph: Backend lowering
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-backend.mlir
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-midend.mlir
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-backend.mlir
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-midend.mlir
    COMMENT "Building subgraph0-backend.mlir (backend lowering)"
    VERBATIM)

# Subgraph: Code generation
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-translate --mlir-to-llvmir
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-backend.mlir
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-backend.mlir
    COMMENT "Building subgraph0.ll (LLVM IR)"
    VERBATIM)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph.o
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
    COMMENT "Building subgraph.o (object file)"
    VERBATIM)

# Staged Executable: Link staged object files with runner
add_executable(transformer-runner-staged
    ${CMAKE_CURRENT_SOURCE_DIR}/transformer_runner.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph.o
)

# Set properties for the staged executable
set_target_properties(transformer-runner-staged PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Include Buddy headers
target_include_directories(transformer-runner-staged PRIVATE
    ${CMAKE_SOURCE_DIR}/frontend/Interfaces
)

# Define paths for the staged executable
target_compile_definitions(transformer-runner-staged PRIVATE
    BUDDY_TRANSFORMER_EXAMPLE_PATH="${CMAKE_CURRENT_SOURCE_DIR}"
    BUDDY_TRANSFORMER_EXAMPLE_BUILD_PATH="${CMAKE_CURRENT_BINARY_DIR}"
)

# Set up library directories and linking
target_link_directories(transformer-runner-staged PRIVATE ${LLVM_LIBRARY_DIR})

# Define libraries to link
set(BUDDY_TRANSFORMER_LIBS
    mlir_c_runner_utils
    omp
)

# Link with required libraries
target_link_libraries(transformer-runner-staged ${BUDDY_TRANSFORMER_LIBS})

# Create targets for building the MLIR files
add_custom_target(buddy-transformer-frontend
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
            ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
    COMMENT "Building frontend MLIR files (PyTorch to TOSA)"
)

add_custom_target(buddy-transformer-midend
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward-midend.mlir
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-midend.mlir
    COMMENT "Building midend optimized MLIR files"
)

add_custom_target(buddy-transformer-backend
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward-backend.mlir
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-backend.mlir
    COMMENT "Building backend lowered MLIR files"
)

add_custom_target(buddy-transformer-codegen
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward.ll
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
            ${CMAKE_CURRENT_BINARY_DIR}/forward.o
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph.o
    COMMENT "Building object files and LLVM IR"
)

add_custom_target(buddy-transformer-executable
    DEPENDS transformer-runner-staged
    COMMENT "Building staged transformer executable (transformer-runner-staged)"
)


# Add dependency to ensure object files are built before staged executable
add_dependencies(transformer-runner-staged buddy-transformer-codegen)


# One-step executable: Link one-step object files with runner
add_executable(transformer-runner
    ${CMAKE_CURRENT_SOURCE_DIR}/transformer_runner.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph-onestep.o
)

# Set properties for the one-step executable
set_target_properties(transformer-runner PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Include Buddy headers for one-step executable
target_include_directories(transformer-runner PRIVATE
    ${CMAKE_SOURCE_DIR}/frontend/Interfaces
)

# Define paths for the one-step executable
target_compile_definitions(transformer-runner PRIVATE
    BUDDY_TRANSFORMER_EXAMPLE_PATH="${CMAKE_CURRENT_SOURCE_DIR}"
    BUDDY_TRANSFORMER_EXAMPLE_BUILD_PATH="${CMAKE_CURRENT_BINARY_DIR}"
)

# Set up library directories and linking for one-step executable
target_link_directories(transformer-runner PRIVATE ${LLVM_LIBRARY_DIR})

# Link with required libraries for one-step executable
target_link_libraries(transformer-runner ${BUDDY_TRANSFORMER_LIBS})

# One-step compilation targets
add_custom_target(buddy-transformer-onestep-codegen
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward-onestep.o
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph-onestep.o
    COMMENT "Building one-step object files"
)

add_custom_target(buddy-transformer-onestep-executable
    DEPENDS transformer-runner
    COMMENT "Building one-step transformer executable"
)

add_custom_target(buddy-transformer
    DEPENDS buddy-transformer-onestep-executable
    COMMENT "Building complete transformer block pipeline (one-step)"
)

# Add dependency for one-step executable
add_dependencies(transformer-runner buddy-transformer-onestep-codegen)

# ===== Timed executable with operator profiling =====
add_executable(transformer-runner-timed
    ${CMAKE_CURRENT_SOURCE_DIR}/transformer_runner.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph-timed.o
)

# Set properties for the timed executable
set_target_properties(transformer-runner-timed PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Include Buddy headers for timed executable
target_include_directories(transformer-runner-timed PRIVATE
    ${CMAKE_SOURCE_DIR}/frontend/Interfaces
)

# Define paths for the timed executable
target_compile_definitions(transformer-runner-timed PRIVATE
    BUDDY_TRANSFORMER_EXAMPLE_PATH="${CMAKE_CURRENT_SOURCE_DIR}"
    BUDDY_TRANSFORMER_EXAMPLE_BUILD_PATH="${CMAKE_CURRENT_BINARY_DIR}"
)

# Set up library directories and linking for timed executable
target_link_directories(transformer-runner-timed PRIVATE ${LLVM_LIBRARY_DIR})

# Link with required libraries for timed executable
target_link_libraries(transformer-runner-timed ${BUDDY_TRANSFORMER_LIBS})

# Timed compilation targets
add_custom_target(buddy-transformer-timed-codegen
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/subgraph-timed.o
    COMMENT "Building timed object file with operator profiling"
)

add_custom_target(buddy-transformer-timed-executable
    DEPENDS transformer-runner-timed
    COMMENT "Building timed transformer executable with operator profiling"
)

# Add dependency for timed executable
add_dependencies(transformer-runner-timed buddy-transformer-timed-codegen)

# Clean target to remove generated files
add_custom_target(buddy-transformer-clean
    COMMAND ${CMAKE_COMMAND} -E remove -f
        ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
        ${CMAKE_CURRENT_BINARY_DIR}/forward-midend.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-midend.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/forward-backend.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-backend.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/forward.ll
        ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
        ${CMAKE_CURRENT_BINARY_DIR}/forward.o
        ${CMAKE_CURRENT_BINARY_DIR}/subgraph.o
        ${CMAKE_CURRENT_BINARY_DIR}/forward-onestep.o
        ${CMAKE_CURRENT_BINARY_DIR}/subgraph-onestep.o
        ${CMAKE_CURRENT_BINARY_DIR}/subgraph-timed.o
        ${CMAKE_CURRENT_BINARY_DIR}/graph.log
        ${CMAKE_CURRENT_BINARY_DIR}/graph_fused.log
    COMMAND ${CMAKE_COMMAND} -E remove -f
        ${CMAKE_CURRENT_BINARY_DIR}/transformer-runner
        ${CMAKE_CURRENT_BINARY_DIR}/transformer-runner-staged
        ${CMAKE_CURRENT_BINARY_DIR}/transformer-runner-timed
    COMMENT "Cleaning BuddyTransformer generated files"
)
