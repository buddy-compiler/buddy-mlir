# ===- CMakeLists.txt ----------------------------------------------------------
#
# CMake configuration for BuddyOneDNN example
#
# ===---------------------------------------------------------------------------

# Environment variables
set(CONDA_PREFIX $ENV{CONDA_PREFIX})
if(NOT CONDA_PREFIX)
    set(CONDA_PREFIX "$ENV{HOME}/miniforge3/envs/buddy")
endif()

set(ONEDNN_INCLUDE ${CONDA_PREFIX}/include)
set(ONEDNN_LIB ${CONDA_PREFIX}/lib)

message(STATUS "CONDA_PREFIX: ${CONDA_PREFIX}")
message(STATUS "oneDNN include: ${ONEDNN_INCLUDE}")
message(STATUS "oneDNN lib: ${ONEDNN_LIB}")

# Find Python3
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ============================================================================
# oneDNN Wrapper Library
# ============================================================================

add_library(onednn_ops SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/onednn_ops.cpp
)

target_include_directories(onednn_ops PRIVATE
    ${ONEDNN_INCLUDE}
)

target_link_directories(onednn_ops PRIVATE
    ${ONEDNN_LIB}
)

target_link_libraries(onednn_ops PRIVATE
    dnnl
)

set_target_properties(onednn_ops PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# Frontend: Generate MLIR from PyTorch
# ============================================================================

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
           ${CMAKE_CURRENT_BINARY_DIR}/graph.log
           ${CMAKE_CURRENT_BINARY_DIR}/graph_transformed.log
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import_simple_model.py
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating MLIR from PyTorch model with oneDNN transform..."
    VERBATIM
)

add_custom_target(buddy-onednn-frontend
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
    COMMENT "Frontend: PyTorch -> MLIR with oneDNN"
)

# ============================================================================
# Midend: MLIR Optimization
# ============================================================================

# Forward graph optimization
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward-opt.mlir
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
              -convert-linalg-to-affine-loops
              -affine-loop-fusion
              -lower-affine
              -cse
              -memref-expand
              -arith-expand
            -o ${CMAKE_CURRENT_BINARY_DIR}/forward-opt.mlir
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
    COMMENT "Optimizing forward.mlir..."
    VERBATIM
)

# Subgraph optimization
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-opt.mlir
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
              -convert-linalg-to-affine-loops
              -affine-loop-fusion
              -lower-affine
              -func-bufferize-dynamic-offset
              -cse
              -memref-expand
              -arith-expand
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-opt.mlir
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
    COMMENT "Optimizing subgraph0.mlir..."
    VERBATIM
)

add_custom_target(buddy-onednn-midend
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward-opt.mlir
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-opt.mlir
    COMMENT "Midend: MLIR optimization"
)

# ============================================================================
# Backend: Lower to LLVM
# ============================================================================

# Forward backend lowering
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward-llvm.mlir
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward-opt.mlir
              -convert-vector-to-llvm
              -convert-arith-to-llvm
              -finalize-memref-to-llvm
              -convert-scf-to-cf
              -convert-cf-to-llvm
              -llvm-request-c-wrappers
              -convert-func-to-llvm
              -reconcile-unrealized-casts
            -o ${CMAKE_CURRENT_BINARY_DIR}/forward-llvm.mlir
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward-opt.mlir
    COMMENT "Lowering forward to LLVM dialect..."
    VERBATIM
)

# Subgraph backend lowering
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-llvm.mlir
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-opt.mlir
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
              -reconcile-unrealized-casts
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-llvm.mlir
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-opt.mlir
    COMMENT "Lowering subgraph0 to LLVM dialect..."
    VERBATIM
)

add_custom_target(buddy-onednn-backend
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward-llvm.mlir
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-llvm.mlir
    COMMENT "Backend: Lower to LLVM dialect"
)

# ============================================================================
# Code Generation: LLVM IR and Object Files
# ============================================================================

# Forward: MLIR -> LLVM IR
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.ll
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-translate --mlir-to-llvmir
            ${CMAKE_CURRENT_BINARY_DIR}/forward-llvm.mlir
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward.ll
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward-llvm.mlir
    COMMENT "Generating forward.ll (LLVM IR)..."
    VERBATIM
)

# Forward: LLVM IR -> Object file
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            ${CMAKE_CURRENT_BINARY_DIR}/forward.ll
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward.o
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward.ll
    COMMENT "Generating forward.o (object file)..."
    VERBATIM
)

# Subgraph: MLIR -> LLVM IR
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-translate --mlir-to-llvmir
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-llvm.mlir
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-llvm.mlir
    COMMENT "Generating subgraph0.ll (LLVM IR)..."
    VERBATIM
)

# Subgraph: LLVM IR -> Object file
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
    COMMENT "Generating subgraph0.o (object file)..."
    VERBATIM
)

add_custom_target(buddy-onednn-codegen
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward.o
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
    COMMENT "Code generation: LLVM IR and object files"
)

# ============================================================================
# Executable: Link everything together
# ============================================================================

# Find MLIR runner utils
set(MLIR_RUNNER_UTILS_LIB "${LLVM_TOOLS_BINARY_DIR}/../lib")

add_executable(buddy-onednn-run
    test_mlir_runner.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
)

target_link_directories(buddy-onednn-run PRIVATE
    ${ONEDNN_LIB}
    ${MLIR_RUNNER_UTILS_LIB}
)

target_link_libraries(buddy-onednn-run
    onednn_ops                      # oneDNN wrapper
    dnnl                            # oneDNN library
    mlir_c_runner_utils             # MLIR C runtime
    mlir_runner_utils               # MLIR runtime
)

set_target_properties(buddy-onednn-run PROPERTIES
    INSTALL_RPATH "${ONEDNN_LIB}:${MLIR_RUNNER_UTILS_LIB}:${CMAKE_CURRENT_BINARY_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH "${ONEDNN_LIB}:${MLIR_RUNNER_UTILS_LIB}:${CMAKE_CURRENT_BINARY_DIR}"
)

# ============================================================================
# All Target
# ============================================================================

add_custom_target(buddy-onednn-all
    DEPENDS buddy-onednn-frontend
            buddy-onednn-midend
            buddy-onednn-backend
            buddy-onednn-codegen
            onednn_ops
            buddy-onednn-run
    COMMENT "Building complete oneDNN integration pipeline"
)

# ============================================================================
# Clean target
# ============================================================================

add_custom_target(buddy-onednn-clean
    COMMAND ${CMAKE_COMMAND} -E remove -f
        ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
        ${CMAKE_CURRENT_BINARY_DIR}/graph.log
        ${CMAKE_CURRENT_BINARY_DIR}/graph_transformed.log
        ${CMAKE_CURRENT_BINARY_DIR}/forward-opt.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-opt.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/forward-llvm.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-llvm.mlir
        ${CMAKE_CURRENT_BINARY_DIR}/forward.ll
        ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.ll
        ${CMAKE_CURRENT_BINARY_DIR}/forward.o
        ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
    COMMENT "Cleaning BuddyOneDNN generated files"
)
