# Define bufferize options as variables to avoid CMake quote parsing issues
set(BUFFERIZE_FULL_OPTS "unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries")
set(BUFFERIZE_SIMPLE_OPTS "bufferize-function-boundaries")
set(TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-f16.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-f16.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-f16.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-f16.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/arg0-f16.data
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
          --precision f16
  COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward-bf16.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-bf16.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/arg0-bf16.data
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
          --precision bf16
  COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
)

add_custom_command(
  OUTPUT forward_prefill.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.mlir
            -simplify-tosa-reshape |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -matmul-vectorization-blis
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=num-threads=48
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill.mlir
  COMMENT "Building forward_prefill.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_prefill.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill.mlir
              -simplify-tosa-reshape |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -matmul-vectorization-blis
            -batchmatmul-optimize
            -batchmatmul-transpose-b-vectorization
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=num-threads=48
            -func-bufferize-dynamic-offset
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill.mlir
    COMMENT "Building subgraph_prefill.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_decode.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.mlir
            -simplify-tosa-reshape |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -matmul-vectorization-blis
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=num-threads=48
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode.mlir
  COMMENT "Building forward_decode.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_decode.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode.mlir
              -simplify-tosa-reshape
              -simplify-tosa-matmul-scalar |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -assume-tight-memref-layout
            -matmul-vectorization-decode=vector-size=128
            -batch-matmul-vectorization-decode=vector-size=128
            -batchmatmul-transpose-b-vectorization=vector-size=16
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=num-threads=48
            -func-bufferize-dynamic-offset
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode.mlir
    COMMENT "Building subgraph_decode.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_prefill-f16.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-f16.mlir
            -simplify-tosa-reshape |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -matmul-vectorization-blis
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=num-threads=48
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-f16.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill-f16.mlir
  COMMENT "Building forward_prefill-f16.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_prefill-f16.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-f16.mlir
              -simplify-tosa-reshape |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -matmul-vectorization-blis
            -batchmatmul-optimize
            -batchmatmul-transpose-b-vectorization
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=num-threads=48
            -func-bufferize-dynamic-offset
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill-f16.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill-f16.mlir
    COMMENT "Building subgraph_prefill-f16.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_decode-f16.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-f16.mlir
            -simplify-tosa-reshape |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -assume-tight-memref-layout
            -matmul-vectorization-blis
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=num-threads=48
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-f16.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode-f16.mlir
  COMMENT "Building forward_decode-f16.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_decode-f16.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-f16.mlir
              -simplify-tosa-reshape
              -simplify-tosa-matmul-scalar |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -assume-tight-memref-layout
            -matmul-vectorization-decode=vector-size=128
            -batch-matmul-vectorization-decode=vector-size=128
            -batchmatmul-transpose-b-vectorization
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=num-threads=48
            -func-bufferize-dynamic-offset
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode-f16.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode-f16.mlir
    COMMENT "Building subgraph_decode-f16.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward-bf16.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward-bf16.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize="bufferize-function-boundaries"
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward-bf16.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward-bf16.mlir
  COMMENT "Building forward.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph-bf16.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-bf16.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp
            -func-bufferize-dynamic-offset
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
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
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph-bf16.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0-bf16.mlir
    COMMENT "Building subgraph.o "
    VERBATIM)

add_library(DEEPSEEKR1 STATIC forward_prefill.o subgraph_prefill.o forward_decode.o subgraph_decode.o)
add_library(DEEPSEEKR1_F16 STATIC forward_prefill-f16.o subgraph_prefill-f16.o forward_decode-f16.o subgraph_decode-f16.o)
add_library(DEEPSEEKR1_BF16 STATIC forward-bf16.o subgraph-bf16.o)

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

SET_TARGET_PROPERTIES(
  DEEPSEEKR1_BF16
  PROPERTIES
  LINKER_LANGUAGE C)

add_executable(buddy-deepseek-r1-run buddy-deepseek-r1-main.cpp)
add_executable(buddy-deepseek-r1-f16-run buddy-deepseek-r1-f16-main.cpp)
add_executable(buddy-deepseek-r1-bf16-run buddy-deepseek-r1-bf16-main.cpp)
add_executable(buddy-deepseek-r1-cli buddy-deepseek-r1-cli.cpp)

set(DEEPSEEKR1_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(DEEPSEEKR1_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(buddy-deepseek-r1-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}/"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}/"
)
target_compile_definitions(buddy-deepseek-r1-f16-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}/"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}/"
)
target_compile_definitions(buddy-deepseek-r1-bf16-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}"
)
target_compile_definitions(buddy-deepseek-r1-cli PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}/"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}/"
)
target_compile_options(buddy-deepseek-r1-cli PRIVATE -fno-rtti)

target_link_directories(buddy-deepseek-r1-run PRIVATE ${LLVM_LIBRARY_DIR})
target_link_directories(buddy-deepseek-r1-f16-run PRIVATE ${LLVM_LIBRARY_DIR})
target_link_directories(buddy-deepseek-r1-bf16-run PRIVATE ${LLVM_LIBRARY_DIR})
target_link_directories(buddy-deepseek-r1-cli PRIVATE ${LLVM_LIBRARY_DIR})

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
set(BUDDY_DEEPSEEKR1_BF16_LIBS
  DEEPSEEKR1_BF16
  mlir_c_runner_utils
  omp
)
if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_DEEPSEEKR1_LIBS mimalloc)
  list(APPEND BUDDY_DEEPSEEKR1_F16_LIBS mimalloc)
  list(APPEND BUDDY_DEEPSEEKR1_BF16_LIBS mimalloc)
endif()

target_link_libraries(buddy-deepseek-r1-run ${BUDDY_DEEPSEEKR1_LIBS})
target_link_libraries(buddy-deepseek-r1-f16-run ${BUDDY_DEEPSEEKR1_F16_LIBS})
target_link_libraries(buddy-deepseek-r1-bf16-run ${BUDDY_DEEPSEEKR1_BF16_LIBS})
target_link_libraries(buddy-deepseek-r1-cli ${BUDDY_DEEPSEEKR1_LIBS} LLVMOption)
