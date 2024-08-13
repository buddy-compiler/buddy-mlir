function(BuildLlama dtype)

add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyLlama/${dtype}-forward.mlir ${BUDDY_EXAMPLES_DIR}/BuddyLlama/${dtype}-subgraph0.mlir
  COMMAND ${Python3_EXECUTABLE} ${BUDDY_EXAMPLES_DIR}/BuddyLlama/import-llama.py --dtype ${dtype}
  COMMENT "Generating forward.mlir, subgraph0.mlir and params.data for ${dtype} model"
)

add_custom_command(
  OUTPUT ${dtype}-forward.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyLlama/${dtype}-forward.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt 
          -arith-expand
          -eliminate-empty-tensors
          -empty-tensor-to-alloc-tensor
          -one-shot-bufferize
          -matmul-parallel-vectorization-optimize
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
        tee ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${dtype}-forward.llvm |
        ${LLVM_TOOLS_BINARY_DIR}/llvm-as -o ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${dtype}-forward.llvm.bc
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/llc ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${dtype}-forward.llvm.bc -filetype=obj -relocation-model=pic -O3 -mattr=+f16c,+avx512bf16
          -o ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${dtype}-forward.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/llc ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${dtype}-forward.llvm.bc -filetype=asm -relocation-model=pic -O3 -mattr=+f16c,+avx512bf16
          -o ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${dtype}-forward.S
  DEPENDS buddy-opt ${BUDDY_EXAMPLES_DIR}/BuddyLlama/${dtype}-forward.mlir
  COMMENT "Building ${dtype}-forward.o"
  VERBATIM)

add_custom_command(
    OUTPUT ${dtype}-subgraph.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyLlama/${dtype}-subgraph0.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-parallel-vectorization-optimize
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
          tee ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${dtype}-subgraph.llvm |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as -o ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${dtype}-subgraph.llvm.bc
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/llc ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${dtype}-subgraph.llvm.bc -filetype=obj -relocation-model=pic -O3 -mattr=+f16c,+avx512bf16
            -o ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${dtype}-subgraph.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/llc ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${dtype}-subgraph.llvm.bc -filetype=asm -relocation-model=pic -O3 -mattr=+f16c,+avx512bf16
            -o ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${dtype}-subgraph.S
    DEPENDS buddy-opt ${BUDDY_EXAMPLES_DIR}/BuddyLlama/${dtype}-subgraph0.mlir
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

add_executable(buddy-${dtype}-llama-run llama-main.cpp llama_utils.cpp)
if (${dtype} STREQUAL "fp32")
  target_compile_definitions(buddy-${dtype}-llama-run PRIVATE LLAMA_FP32_TYPE)
elseif (${dtype} STREQUAL "fp16")
  target_compile_definitions(buddy-${dtype}-llama-run PRIVATE LLAMA_FP16_TYPE)
elseif (${dtype} STREQUAL "bf16")
  target_compile_definitions(buddy-${dtype}-llama-run PRIVATE LLAMA_BF16_TYPE)
endif()

target_link_directories(buddy-${dtype}-llama-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_include_directories(buddy-${dtype}-llama-run PRIVATE ${BUDDY_EXAMPLES_DIR}/BuddyLlama/FP16/include)

set(BUDDY_LLAMA_LIBS
  ${dtype}Llama
  mlir_c_runner_utils
  omp
)
if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_LLAMA_LIBS mimalloc)
endif()

target_link_libraries(buddy-${dtype}-llama-run ${BUDDY_LLAMA_LIBS})
target_compile_definitions(buddy-${dtype}-llama-run PRIVATE LLAMA_DTYPE=${dtype})
endfunction()

BuildLlama("fp32")
BuildLlama("fp16")
BuildLlama("bf16")
