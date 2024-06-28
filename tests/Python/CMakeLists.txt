add_custom_command(
  OUTPUT ${BUDDY_SOURCE_DIR}/tests/Python/test_embedding.mlir
  COMMAND ${Python3_EXECUTABLE} ${BUDDY_SOURCE_DIR}/tests/Python/test_embedding_quantize.py
  COMMENT "Generating test_embedding.mlir..."
)

add_custom_command(
  OUTPUT ${BUDDY_SOURCE_DIR}/tests/Python/test_mul.mlir
  COMMAND ${Python3_EXECUTABLE} ${BUDDY_SOURCE_DIR}/tests/Python/test_arith_mul_quantize.py
  COMMENT "Generating test_mul.mlir..."
)

add_custom_command(
  OUTPUT test_embedding.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_SOURCE_DIR}/tests/Python/test_embedding.mlir
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
          -o ${BUDDY_BINARY_DIR}/../tests/Python/test_embedding.o
  DEPENDS buddy-opt ${BUDDY_SOURCE_DIR}/tests/Python/test_embedding.mlir
  COMMENT "Building test_embedding.o "
  VERBATIM)

add_custom_command(
  OUTPUT test_mul.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_SOURCE_DIR}/tests/Python/test_mul.mlir
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
       ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llvm-as |
        ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
          -o ${BUDDY_BINARY_DIR}/../tests/Python/test_mul.o
  DEPENDS buddy-opt ${BUDDY_SOURCE_DIR}/tests/Python/test_mul.mlir
  COMMENT "Building test_mul.o "
  VERBATIM)

add_library(Testfloat_embedding STATIC test_embedding.o)
add_library(Testfloat_mul STATIC test_mul.o)


SET_TARGET_PROPERTIES(
  Testfloat_mul
  Testfloat_embedding
  PROPERTIES
  LINKER_LANGUAGE C)

set(BUDDY_TESTFLOAT_LIBS
  Testfloat_mul
  Testfloat_embedding
  mlir_c_runner_utils
  omp
)

add_executable(buddy-testfloat16-embedding-run testfloat16_embedding.cpp)
add_executable(buddy-testfloat16-run testfloat16_mul.cpp)

target_link_directories(buddy-testfloat16-embedding-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_link_libraries(buddy-testfloat16-embedding-run ${BUDDY_TESTFLOAT_LIBS})
target_link_libraries(buddy-testfloat16-run ${BUDDY_TESTFLOAT_LIBS})

