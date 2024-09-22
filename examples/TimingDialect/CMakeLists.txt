add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/TimingDialect/forward.mlir ${BUDDY_EXAMPLES_DIR}/TimingDialect/subgraph0.mlir ${BUDDY_EXAMPLES_DIR}/TimingDialect/arg0.data
  COMMAND ${Python3_EXECUTABLE} ${BUDDY_EXAMPLES_DIR}/TimingDialect/import-llama2.py
  COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
)

add_custom_command(
  OUTPUT forward.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_EXAMPLES_DIR}/TimingDialect/forward.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            --matmul-parallel-vectorization-optimize
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
        ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
        ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
          -o ${BUDDY_BINARY_DIR}/../examples/TimingDialect/forward.o
  DEPENDS buddy-opt ${BUDDY_EXAMPLES_DIR}/TimingDialect/forward.mlir
  COMMENT "Building forward.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_EXAMPLES_DIR}/TimingDialect/subgraph0.mlir --timing |
            ${BUDDY_BINARY_DIR}/buddy-opt  
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            --matmul-parallel-vectorization-optimize
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
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${BUDDY_BINARY_DIR}/../examples/TimingDialect/subgraph.o
    DEPENDS buddy-opt ${BUDDY_EXAMPLES_DIR}/TimingDialect/subgraph0.mlir
    COMMENT "Building subgraph.o "
    VERBATIM)

add_library(LLAMA_T STATIC forward.o subgraph.o)

add_library(TIMING_UTILS STATIC utils.cpp runtime.cpp)  

SET_SOURCE_FILES_PROPERTIES(
  template.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true)

SET_TARGET_PROPERTIES(
  LLAMA_T
  PROPERTIES
  LINKER_LANGUAGE C)

add_executable(buddy-llama-run-timing llama-main.cpp)
target_link_directories(buddy-llama-run-timing PRIVATE ${LLVM_MLIR_LIBRARY_DIR})

set(BUDDY_LLAMA_T_LIBS
  LLAMA_T
  TIMING_UTILS
  mlir_c_runner_utils
  mlir_runner_utils
  omp
)

if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_LLAMA_T_LIBS mimalloc)
endif()

target_link_libraries(buddy-llama-run-timing ${BUDDY_LLAMA_T_LIBS})
