function(BuildLlama mtype dtype)

add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyLlama/${mtype}-${dtype}-forward.mlir ${BUDDY_EXAMPLES_DIR}/BuddyLlama/${mtype}-${dtype}-subgraph0.mlir
  COMMAND ${Python3_EXECUTABLE} ${BUDDY_EXAMPLES_DIR}/BuddyLlama/import-llama.py --mtype ${mtype} --dtype ${dtype}
  COMMENT "Generating forward.mlir, subgraph0.mlir and params.data for ${mtype}-${dtype} model"
)

add_custom_command(
  OUTPUT ${mtype}-${dtype}-forward.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyLlama/${mtype}-${dtype}-forward.mlir 
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
        tee ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${mtype}-${dtype}-forward.llvm |
        ${LLVM_MLIR_BINARY_DIR}/llvm-as -o ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${mtype}-${dtype}-forward.llvm.bc
  COMMAND ${LLVM_MLIR_BINARY_DIR}/llc ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${mtype}-${dtype}-forward.llvm.bc -filetype=obj -relocation-model=pic -O3 -mattr=+f16c,+avx512bf16
          -o ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${mtype}-${dtype}-forward.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/llc ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${mtype}-${dtype}-forward.llvm.bc -filetype=asm -relocation-model=pic -O3 -mattr=+f16c,+avx512bf16
          -o ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${mtype}-${dtype}-forward.S
  DEPENDS buddy-opt ${BUDDY_EXAMPLES_DIR}/BuddyLlama/${mtype}-${dtype}-forward.mlir
  COMMENT "Building ${mtype}-${dtype}-forward.o"
  VERBATIM)

add_custom_command(
    OUTPUT ${mtype}-${dtype}-subgraph.o
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyLlama/${mtype}-${dtype}-subgraph0.mlir 
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
          tee ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${mtype}-${dtype}-subgraph.llvm |
          ${LLVM_MLIR_BINARY_DIR}/llvm-as -o ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${mtype}-${dtype}-subgraph.llvm.bc
    COMMAND ${LLVM_MLIR_BINARY_DIR}/llc ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${mtype}-${dtype}-subgraph.llvm.bc -filetype=obj -relocation-model=pic -O3 -mattr=+f16c,+avx512bf16
            -o ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${mtype}-${dtype}-subgraph.o
    COMMAND ${LLVM_MLIR_BINARY_DIR}/llc ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${mtype}-${dtype}-subgraph.llvm.bc -filetype=asm -relocation-model=pic -O3 -mattr=+f16c,+avx512bf16
            -o ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/${mtype}-${dtype}-subgraph.S
    DEPENDS buddy-opt ${BUDDY_EXAMPLES_DIR}/BuddyLlama/${mtype}-${dtype}-subgraph0.mlir
    COMMENT "Building ${mtype}-${dtype}-subgraph.o"
    VERBATIM)

add_library(${mtype}-${dtype}Llama STATIC ${mtype}-${dtype}-forward.o ${mtype}-${dtype}-subgraph.o)

SET_SOURCE_FILES_PROPERTIES(
  template.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true)

SET_TARGET_PROPERTIES(
  ${mtype}-${dtype}Llama
  PROPERTIES
  LINKER_LANGUAGE C)

if(dtype STREQUAL fp32)
add_executable(buddy-${mtype}-${dtype}-llama-run f32-llama-main.cpp)
elseif(dtype STREQUAL fp16)
add_executable(buddy-${mtype}-${dtype}-llama-run f16-llama-main.cpp)
else()
add_executable(buddy-${mtype}-${dtype}-llama-run bf16-llama-main.cpp)
endif()

target_link_directories(buddy-${mtype}-${dtype}-llama-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_include_directories(buddy-${mtype}-${dtype}-llama-run PRIVATE ${BUDDY_EXAMPLES_DIR}/BuddyLlama/FP16/include)

set(BUDDY_LLAMA_LIBS
  ${mtype}-${dtype}Llama
  mlir_c_runner_utils
  omp
)
if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_LLAMA_LIBS mimalloc)
endif()

target_link_libraries(buddy-${mtype}-${dtype}-llama-run ${BUDDY_LLAMA_LIBS})
target_compile_definitions(buddy-${mtype}-${dtype}-llama-run PRIVATE LLAMA_DTYPE=${dtype})
endfunction()

BuildLlama("1.1b" "fp32")
BuildLlama("7b" "fp32")
BuildLlama("1.1b" "fp16")
BuildLlama("7b" "fp16")
BuildLlama("1.1b" "bf16")
BuildLlama("7b" "bf16")
