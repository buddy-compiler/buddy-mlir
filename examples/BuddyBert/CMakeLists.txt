add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
         ${CMAKE_CURRENT_BINARY_DIR}/arg1.data
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-bert.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)



add_custom_command(
  OUTPUT forward.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt  ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), eliminate-empty-tensors, func.func(llvm-request-c-wrappers),convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf, convert-cf-to-llvm, convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0.o
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
        ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O0
          -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
  COMMENT "Building subgraph.o "
  VERBATIM)

add_library(BERT STATIC forward.o subgraph0.o)

SET_TARGET_PROPERTIES(BERT PROPERTIES LINKER_LANGUAGE C)

add_executable(buddy-bert-run bert-main.cpp)

set(BERT_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(BERT_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(buddy-bert-run PRIVATE
  BERT_EXAMPLE_PATH="${BERT_EXAMPLE_PATH}"
  BERT_EXAMPLE_BUILD_PATH="${BERT_EXAMPLE_BUILD_PATH}"
)

target_link_directories(buddy-bert-run PRIVATE ${LLVM_LIBRARY_DIR})

set(BUDDY_BERT_LIBS BERT mlir_c_runner_utils)
target_link_libraries(buddy-bert-run ${BUDDY_BERT_LIBS})
