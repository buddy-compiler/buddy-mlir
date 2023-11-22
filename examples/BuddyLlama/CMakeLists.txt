add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyLlama/llama.mlir ${BUDDY_EXAMPLES_DIR}/BuddyLlama/arg0.data
  COMMAND python ${BUDDY_EXAMPLES_DIR}/BuddyLlama/import-llama2.py
  COMMENT "Generating llama.mlir and arg0.data..."
)

set(OPT_FLAGS [[buddy-opt -matmul-paralell-vectorization-optimize='vector-size=128 serial-store-opt=1']])
set(MAX_NESTED [[buddy-opt -affine-parallelize='max-nested=2']])

if (${BUDDY_LLAMA_USE_LTO})
  set(CMAKE_CXX_COMPILER ${LLVM_MLIR_BINARY_DIR}/clang++ CACHE PATH "clang++" FORCE)
  add_custom_command(
    OUTPUT llama.o
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyLlama/llama.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -arith-expand
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
              -linalg-bufferize |
            sh -c ${BUDDY_BINARY_DIR}/${OPT_FLAGS} | # Quite hacky way to avoid the escape problem.
            ${BUDDY_BINARY_DIR}/buddy-opt
              -batchmatmul-optimize
              -convert-linalg-to-affine-loops
              -affine-loop-fusion |
            sh -c ${BUDDY_BINARY_DIR}/${MAX_NESTED} |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -lower-affine
              -scf-parallel-loop-fusion
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
          ${CMAKE_CXX_COMPILER} -c -x ir -O3 -march=native -fpic -flto
            -o ${BUDDY_BINARY_DIR}/../examples/BuddyLlama/llama.o -
    DEPENDS buddy-opt ${BUDDY_EXAMPLES_DIR}/BuddyLlama/llama.mlir
    COMMENT "Building llama.o "
    VERBATIM)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fpic -flto -fuse-ld=lld -O3")
else()
  add_custom_command(
    OUTPUT llama.o
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyLlama/llama.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -arith-expand
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
              -linalg-bufferize |
            sh -c ${BUDDY_BINARY_DIR}/${OPT_FLAGS} | # Quite hacky way to avoid the escape problem.
            ${BUDDY_BINARY_DIR}/buddy-opt
              -batchmatmul-optimize
              -convert-linalg-to-affine-loops
              -affine-loop-fusion |
            sh -c ${BUDDY_BINARY_DIR}/${MAX_NESTED} |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -lower-affine
              -scf-parallel-loop-fusion
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
endif()

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
if(BUDDY_MLIR_USE_SNMALLOC)
  list(APPEND BUDDY_LLAMA_LIBS snmalloc-static)
endif()

target_link_libraries(buddy-llama-run ${BUDDY_LLAMA_LIBS})
