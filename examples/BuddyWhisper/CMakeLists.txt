add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-whisper.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
)
set(PATTERN_ARG "test-decompose-pad-tensor")
add_custom_command(
  OUTPUT forward.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg)" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops),matmul-parallel-vectorization-optimize, batchmatmul-optimize, eliminate-empty-tensors,func-bufferize-dynamic-offset, func.func(llvm-request-c-wrappers),convert-scf-to-openmp, convert-openmp-to-llvm, convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf,  convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
    OUTPUT subgraph0.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -test-linalg-transform-patterns=${PATTERN_ARG} |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -arith-expand
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -empty-tensor-to-alloc-tensor
              -one-shot-bufferize="bufferize-function-boundaries"
              -ownership-based-buffer-deallocation
              -buffer-deallocation-simplification
              -bufferization-lower-deallocations
              -matmul-parallel-vectorization-optimize
              -batchmatmul-optimize
              -convert-linalg-to-affine-loops
              -affine-loop-fusion
              -affine-parallelize
              -lower-affine
              -convert-scf-to-openmp
              -func-bufferize-dynamic-offset
              -convert-linalg-to-loops
              -convert-vector-to-scf
              -expand-strided-metadata
              -lower-affine
              -cse
              -convert-vector-to-llvm
              -memref-expand
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
            ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O3 -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
    COMMENT "Building subgraph0.o "
    VERBATIM)

add_library(WHISPER STATIC forward.o subgraph0.o)

SET_SOURCE_FILES_PROPERTIES(
  template.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true)

SET_TARGET_PROPERTIES(
  WHISPER
  PROPERTIES
  LINKER_LANGUAGE C)

set(BUDDY_WHISPER_FILES
  whisper-main.cpp
)

add_executable(buddy-whisper-run ${BUDDY_WHISPER_FILES})

set(WHISPER_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(WHISPER_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(buddy-whisper-run PRIVATE
  WHISPER_EXAMPLE_PATH="${WHISPER_EXAMPLE_PATH}"
  WHISPER_EXAMPLE_BUILD_PATH="${WHISPER_EXAMPLE_BUILD_PATH}"
)

target_link_directories(buddy-whisper-run PRIVATE ${LLVM_LIBRARY_DIR})

set(BUDDY_WHISPER_LIBS
  WHISPER
  BuddyLibDAP
  mlir_c_runner_utils
  omp
)
if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_WHISPER_LIBS mimalloc)
endif()

target_link_libraries(buddy-whisper-run ${BUDDY_WHISPER_LIBS})
