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
            -affine-loop-fusion
            -affine-parallelize
            -convert-scf-to-openmp
            -convert-vector-to-scf
            -expand-strided-metadata
            -lower-affine
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward-midend.mlir
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  COMMENT "Optimizing forward.mlir (midend passes)"
  VERBATIM)

# Backend: Optimize forward.mlir (Stage 2 - Backend optimization)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward-optimized.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward-midend.mlir
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
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
            -affine-loop-fusion
            -affine-parallelize
            -convert-scf-to-openmp
            -func-bufferize-dynamic-offset
            -convert-vector-to-scf
            -expand-strided-metadata
            -lower-affine
            -cse
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-midend.mlir
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
    COMMENT "Optimizing subgraph0.mlir (midend passes)"
    VERBATIM)

# Backend: Optimize subgraph0.mlir (Stage 2 - Backend optimization)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-optimized.mlir
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-midend.mlir
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
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

add_custom_target(buddy-attention
    DEPENDS buddy-attention-backend
    COMMENT "Building all optimized attention MLIR files"
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
        ${CMAKE_CURRENT_BINARY_DIR}/graph.log
        ${CMAKE_CURRENT_BINARY_DIR}/graph_fused.log
    COMMENT "Cleaning BuddyAttention generated files"
)
