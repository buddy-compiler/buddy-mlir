add_custom_command(
  OUTPUT forward.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyGraph/forward.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-paralell-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
            -func-bufferize
            -arith-bufferize
            -tensor-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -convert-vector-to-scf
            -expand-strided-metadata
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llvm-as |
        ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
          -o ${BUDDY_BINARY_DIR}/../examples/BuddyGraph/forward.o
  COMMENT "Building forward.o "
  VERBATIM)

add_custom_target(forward
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyGraph/forward.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -arith-expand
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
              -one-shot-bufferize
              -matmul-paralell-vectorization-optimize
              -batchmatmul-optimize
              -convert-linalg-to-affine-loops
              -affine-loop-fusion
              -affine-parallelize
              -lower-affine
              -convert-scf-to-openmp
              -func-bufferize
              -arith-bufferize
              -tensor-bufferize
              -buffer-deallocation
              -finalizing-bufferize
              -convert-vector-to-scf
              -expand-strided-metadata
              -convert-vector-to-llvm
              -memref-expand
              -arith-expand
              -convert-arith-to-llvm
              -finalize-memref-to-llvm
              -convert-scf-to-cf
              -llvm-request-c-wrappers
              -convert-openmp-to-llvm
              -convert-arith-to-llvm
              -convert-math-to-llvm
              -convert-math-to-libm 
              -convert-func-to-llvm
              -reconcile-unrealized-casts 
            -o ${BUDDY_BINARY_DIR}/../examples/BuddyGraph/forward.mlir
    COMMENT "Building forward.mlir "
    VERBATIM)

add_custom_command(
    OUTPUT subgraph.o
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyGraph/subgraph0.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-paralell-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
            -func-bufferize-dynamic-offset
            -tensor-bufferize
            -arith-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -convert-vector-to-scf
            -expand-strided-metadata
            -cse
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llvm-as |
          ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${BUDDY_BINARY_DIR}/../examples/BuddyGraph/subgraph.o
    COMMENT "Building subgraph.o "
    VERBATIM)

add_library(LLAMA-Graph STATIC forward.o subgraph.o)

SET_SOURCE_FILES_PROPERTIES(
  template.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true)

SET_TARGET_PROPERTIES(
  LLAMA-Graph
  PROPERTIES
  LINKER_LANGUAGE C)

add_executable(buddy-llama-graph-run llama-main.cpp)
target_link_directories(buddy-llama-graph-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR})

set(BUDDY_LLAMA_GRAPH_LIBS
  LLAMA-Graph
  mlir_c_runner_utils
  omp
)
if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_LLAMA_GRAPH_LIBS mimalloc)
endif()

target_link_libraries(buddy-llama-graph-run ${BUDDY_LLAMA_GRAPH_LIBS})
