add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward-f16.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-f16.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/arg0-f16.data
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
          --precision f16
  COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
)

add_custom_command(
  OUTPUT forward.o
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
            -reconcile-unrealized-casts |
        ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
        ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  COMMENT "Building forward.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph.o
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
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
    COMMENT "Building subgraph.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward-f16.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward-f16.mlir
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
            -reconcile-unrealized-casts |
        ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
        ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward-f16.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward-f16.mlir
  COMMENT "Building forward.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph-f16.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-f16.mlir
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
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph-f16.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-f16.mlir
    COMMENT "Building subgraph.o "
    VERBATIM)

add_library(DEEPSEEKR1 STATIC forward.o subgraph.o)
add_library(DEEPSEEKR1_F16 STATIC forward-f16.o subgraph-f16.o)

SET_SOURCE_FILES_PROPERTIES(
  template.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true)

SET_TARGET_PROPERTIES(
  DEEPSEEKR1
  PROPERTIES
  LINKER_LANGUAGE C)

SET_TARGET_PROPERTIES(
  DEEPSEEKR1_F16
  PROPERTIES
  LINKER_LANGUAGE C)

add_executable(buddy-deepseek-r1-run buddy-deepseek-r1-main.cpp)
add_executable(buddy-deepseek-r1-f16-run buddy-deepseek-r1-f16-main.cpp)

set(DEEPSEEKR1_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(DEEPSEEKR1_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(buddy-deepseek-r1-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}"
)
target_compile_definitions(buddy-deepseek-r1-f16-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}"
)

target_link_directories(buddy-deepseek-r1-run PRIVATE ${LLVM_LIBRARY_DIR})
target_link_directories(buddy-deepseek-r1-f16-run PRIVATE ${LLVM_LIBRARY_DIR})

set(BUDDY_DEEPSEEKR1_LIBS
  DEEPSEEKR1
  mlir_c_runner_utils
  omp
)
set(BUDDY_DEEPSEEKR1_F16_LIBS
  DEEPSEEKR1_F16
  mlir_c_runner_utils
  omp
)
if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_DEEPSEEKR1_LIBS mimalloc)
  list(APPEND BUDDY_DEEPSEEKR1_F16_LIBS mimalloc)
endif()

target_link_libraries(buddy-deepseek-r1-run ${BUDDY_DEEPSEEKR1_LIBS})
target_link_libraries(buddy-deepseek-r1-f16-run ${BUDDY_DEEPSEEKR1_F16_LIBS})
