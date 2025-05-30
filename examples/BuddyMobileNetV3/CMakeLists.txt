add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
         ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/buddy-mobilenetv3-import.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)


add_custom_command(
  OUTPUT forward.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
            -pass-pipeline
            "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), \
            empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline
            "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), \
            eliminate-empty-tensors, func.func(llvm-request-c-wrappers), \
            convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf, convert-cf-to-llvm, \
            convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, \
            convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)


add_custom_command(
  OUTPUT subgraph0.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
            -pass-pipeline
            "builtin.module(func.func(tosa-to-linalg-named, tosa-to-arith, tosa-to-linalg, tosa-to-tensor))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -convert-elementwise-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -func-bufferize-dynamic-offset
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -expand-strided-metadata
            -lower-affine
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
          buddy-opt
  COMMENT "Building subgraph0.o"
  VERBATIM)

add_library(MOBILENETV3 STATIC subgraph0.o forward.o)

SET_TARGET_PROPERTIES(MOBILENETV3 PROPERTIES LINKER_LANGUAGE C)

add_executable(buddy-mobilenetv3-run buddy-mobilenetv3-main.cpp)

set(MOBILENETV3_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(MOBILENETV3_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(buddy-mobilenetv3-run PRIVATE
  MOBILENETV3_EXAMPLE_PATH="${MOBILENETV3_EXAMPLE_PATH}"
  MOBILENETV3_EXAMPLE_BUILD_PATH="${MOBILENETV3_EXAMPLE_BUILD_PATH}"
)

target_link_directories(buddy-mobilenetv3-run PRIVATE ${LLVM_LIBRARY_DIR})

set(BUDDY_MOBILENETV3_LIBS MOBILENETV3 mlir_c_runner_utils BuddyLibDIP ${PNG_LIBRARIES})
target_link_libraries(buddy-mobilenetv3-run ${BUDDY_MOBILENETV3_LIBS})
