add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyF16Llama/f32-forward.mlir ${BUDDY_EXAMPLES_DIR}/BuddyF16Llama/f32-subgraph0.mlir ${BUDDY_EXAMPLES_DIR}/BuddyF16Llama/f32-params.data
  ${BUDDY_EXAMPLES_DIR}/BuddyF16Llama/fp16-forward.mlir ${BUDDY_EXAMPLES_DIR}/BuddyF16Llama/fp16-subgraph0.mlir ${BUDDY_EXAMPLES_DIR}/BuddyF16Llama/fp16-params.data
  COMMAND ${Python3_EXECUTABLE} ${BUDDY_EXAMPLES_DIR}/BuddyF16Llama/import-llama2.py
  COMMENT "Generating forward.mlir, subgraph0.mlir and params.data for f32 and fp16 models"
)

function(BuildLlama dtype)
add_custom_command(
  OUTPUT ${dtype}-forward.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyF16Llama/${dtype}-forward.mlir 
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
          -o ${BUDDY_BINARY_DIR}/../examples/BuddyF16Llama/${dtype}-forward.o
  DEPENDS buddy-opt ${BUDDY_EXAMPLES_DIR}/BuddyF16Llama/${dtype}-forward.mlir
  COMMENT "Building ${dtype}-forward.o"
  VERBATIM)

add_custom_command(
    OUTPUT ${dtype}-subgraph.o
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyF16Llama/${dtype}-subgraph0.mlir 
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
          ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3 -mattr=-sse2
            -o ${BUDDY_BINARY_DIR}/../examples/BuddyF16Llama/${dtype}-subgraph.o
    DEPENDS buddy-opt ${BUDDY_EXAMPLES_DIR}/BuddyF16Llama/${dtype}-subgraph0.mlir
    COMMENT "Building ${dtype}-subgraph.o"
    VERBATIM)

add_library(${dtype}Llama STATIC ${dtype}-forward.o ${dtype}-subgraph.o)

SET_SOURCE_FILES_PROPERTIES(
  template.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true)

SET_TARGET_PROPERTIES(
  ${dtype}Llama
  PROPERTIES
  LINKER_LANGUAGE C)

add_executable(buddy-${dtype}-llama-run ${dtype}-llama-main.cpp fp16_h2f.cpp fp16_f2h.cpp)
target_link_directories(buddy-${dtype}-llama-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_include_directories(buddy-${dtype}-llama-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR}/../../../)

set(BUDDY_LLAMA_LIBS
  ${dtype}Llama
  mlir_c_runner_utils
  omp
)
if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_LLAMA_LIBS mimalloc)
endif()

target_link_libraries(buddy-${dtype}-llama-run ${BUDDY_LLAMA_LIBS})
endfunction()

BuildLlama(f32)
BuildLlama(fp16)