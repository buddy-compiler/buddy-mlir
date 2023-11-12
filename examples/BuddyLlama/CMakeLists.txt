add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyLlama/llama.mlir ${BUDDY_EXAMPLES_DIR}/BuddyLlama/arg0.data
  COMMAND python ${BUDDY_EXAMPLES_DIR}/BuddyLlama/import-llama2.py
  COMMENT "Generating llama.mlir and arg0.data..."
)

add_custom_command(
  OUTPUT llama.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyLlama/llama.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -linalg-bufferize
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
          -o ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/llama.o
  DEPENDS buddy-opt ${BUDDY_EXAMPLES_DIR}/BuddyLlama/llama.mlir
  COMMENT "Building llama.o "
  VERBATIM)

add_library(LLAMA STATIC llama.o)

  SET_SOURCE_FILES_PROPERTIES(
    template.o
    PROPERTIES
    EXTERNAL_OBJECT true
    GENERATED true)
  
  SET_TARGET_PROPERTIES(
    LLAMA
    PROPERTIES
    LINKER_LANGUAGE C)
  
  add_executable(buddy-llama-run llama-main.cpp)
  target_link_directories(buddy-llama-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
  
  set(BUDDY_LLAMA_LIBS
    LLAMA
    mlir_c_runner_utils
    omp
  )
  if(BUDDY_MLIR_USE_MIMALLOC)
    list(APPEND BUDDY_LLAMA_LIBS mimalloc)
  endif()
  
  target_link_libraries(buddy-llama-run ${BUDDY_LLAMA_LIBS})
