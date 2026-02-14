# Define bufferize options as variables to avoid CMake quote parsing issues
set(BUFFERIZE_FULL_OPTS "unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries")
set(BUFFERIZE_SIMPLE_OPTS "bufferize-function-boundaries")
set(TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")

set(OPENMP_NUM_THREADS "num-threads=48")
set(LLC_RISCV_ATTRS "")
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_decode0.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/forward_decode1.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/forward_decode2.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/forward_decode3.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/forward_decode5.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/forward_decode169.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode0.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode1.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode2.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode3.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode5.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode169.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill0.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill1.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill2.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill3.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill5.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill169.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill0.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill1.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill2.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill3.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill5.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill169.mlir
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1.py 
          --output-dir ${CMAKE_CURRENT_BINARY_DIR} 
  COMMENT "Generating forward.mlir, subgraph.mlir and arg0.data..."
)


set(FORWARD_PASSES
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
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-ub-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
)

set(SUBGRAPH_PASSES
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
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-ub-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
)

set(SUBGRAPH_DECODE_PASSES
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -assume-tight-memref-layout
            -staticize-memref-layout
            -matmul-vectorization-decode=vector-size=128
            -batch-matmul-vectorization-decode=vector-size=128
            -batchmatmul-transpose-b-vectorization=vector-size=16
            -convert-linalg-to-affine-loops
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-ub-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts

)


# string(REPLACE ";" " " FORWARD_PASSES_STR "${FORWARD_PASSES_LIST}")
# string(REPLACE ";" " " SUBGRAPH_PASSES_STR "${SUBGRAPH_PASSES}")

function(build_forward_prefill N)
  add_custom_command(
    OUTPUT forward_prefill${N}.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill${N}.mlir
            -simplify-tosa-reshape |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt -pass-pipeline ${TOSA_PIPELINE}
            | ${BUDDY_BINARY_DIR}/buddy-opt ${FORWARD_PASSES}
            | ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir
            | ${LLVM_TOOLS_BINARY_DIR}/llvm-as
            | ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
              -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill${N}.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill${N}.mlir
    COMMENT "Building forward_prefill${N}.o"
    VERBATIM)
endfunction()

function(build_subgraph_prefill N)
  add_custom_command(
    OUTPUT subgraph0_prefill${N}.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill${N}.mlir
            -simplify-tosa-reshape |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt -pass-pipeline ${TOSA_PIPELINE}
            | ${BUDDY_BINARY_DIR}/buddy-opt ${SUBGRAPH_PASSES}
            | ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir
            | ${LLVM_TOOLS_BINARY_DIR}/llvm-as
            | ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
              -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill${N}.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill${N}.mlir
    COMMENT "Building subgraph${N}_prefill.o"
    VERBATIM)
endfunction()

foreach(N 0 1 2 3 5 169)
  build_forward_prefill(${N})
  build_subgraph_prefill(${N})
  add_library(DSR1TP_prefill${N} STATIC forward_prefill${N}.o subgraph0_prefill${N}.o)
endforeach()


function(build_forward_decode N)
  add_custom_command(
    OUTPUT forward_decode${N}.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode${N}.mlir
            -simplify-tosa-reshape |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt -pass-pipeline ${TOSA_PIPELINE}
            | ${BUDDY_BINARY_DIR}/buddy-opt ${FORWARD_PASSES}
            | ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir
            | ${LLVM_TOOLS_BINARY_DIR}/llvm-as
            | ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
              -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode${N}.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode${N}.mlir
    COMMENT "Building forward_decode${N}.o"
    VERBATIM)
endfunction()

function(build_subgraph_decode N)
  add_custom_command(
    OUTPUT subgraph0_decode${N}.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode${N}.mlir
            -simplify-tosa-reshape -simplify-tosa-matmul-scalar |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt -pass-pipeline ${TOSA_PIPELINE}
            | ${BUDDY_BINARY_DIR}/buddy-opt ${SUBGRAPH_DECODE_PASSES}
            | ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir
            | ${LLVM_TOOLS_BINARY_DIR}/llvm-as
            | ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
              -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode${N}.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode${N}.mlir
    COMMENT "Building subgraph${N}_decode.o"
    VERBATIM)
endfunction()

foreach(N 0 1 2 3 5 169)
  build_forward_decode(${N})
  build_subgraph_decode(${N})
  add_library(DSR1TP_decode${N} STATIC forward_decode${N}.o subgraph0_decode${N}.o)
endforeach()


SET_SOURCE_FILES_PROPERTIES(
  template.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true)

SET_TARGET_PROPERTIES(
  DSR1TP_prefill0
  DSR1TP_prefill1
  DSR1TP_prefill2
  DSR1TP_prefill3
  DSR1TP_prefill5
  DSR1TP_prefill169
  DSR1TP_decode0
  DSR1TP_decode1
  DSR1TP_decode2
  DSR1TP_decode3
  DSR1TP_decode5
  DSR1TP_decode169
  PROPERTIES
  LINKER_LANGUAGE C)

add_executable(buddy-dis-run dis-main.cpp)

set(DSR1TP_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(DSR1TP_EXAMPLE_BUILD_PATH "${CMAKE_CURRENT_BINARY_DIR}")

target_compile_definitions(buddy-dis-run PRIVATE 
  DSR1TP_EXAMPLE_PATH="${DSR1TP_EXAMPLE_PATH}" 
  DSR1TP_EXAMPLE_BUILD_PATH="${DSR1TP_EXAMPLE_BUILD_PATH}"
)

target_link_directories(buddy-dis-run PRIVATE ${LLVM_LIBRARY_DIR})

set(BUDDY_DIS_LLAMA_LIBS
  DSR1TP_prefill0
  DSR1TP_prefill1
  DSR1TP_prefill2
  DSR1TP_prefill3
  DSR1TP_prefill5
  DSR1TP_prefill169
  DSR1TP_decode0
  DSR1TP_decode1
  DSR1TP_decode2
  DSR1TP_decode3
  DSR1TP_decode5
  DSR1TP_decode169
  mlir_c_runner_utils
  omp
)

target_link_libraries(buddy-dis-run ${BUDDY_DIS_LLAMA_LIBS})

