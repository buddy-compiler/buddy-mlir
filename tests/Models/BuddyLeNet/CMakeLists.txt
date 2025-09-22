add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_LeNet.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)


add_custom_command(
  OUTPUT forward.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
            -tosa-to-linalg-named
            -tosa-to-linalg
            -tosa-to-tensor
            -tosa-to-arith
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -buffer-deallocation-simplification
            -convert-linalg-to-loops
            -eliminate-empty-tensor
            -llvm-request-c-wrappers
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-scf-to-cf
            -convert-arith-to-llvm
            -expand-strided-metadata
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
            -tosa-to-linalg-named
            -tosa-to-linalg
            -tosa-to-tensor
            -tosa-to-arith |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -func-bufferize-dynamic-offset
            -convert-vector-to-scf
            -expand-strided-metadata
            -lower-affine
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM)

set(LENET_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(LENET_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})

add_library(LENET_MODELS_CPU STATIC subgraph0.o forward.o)

SET_TARGET_PROPERTIES(LENET_MODELS_CPU PROPERTIES LINKER_LANGUAGE C)

add_executable(buddy-lenet-run-test-cpu LeNetInferTest.cpp)
target_link_directories(buddy-lenet-run-test-cpu PRIVATE ${LLVM_LIBRARY_DIR})

if(NOT DEFINED BUDDY_ENABLE_PNG)
  message(FATAL_ERROR "To run LeNet inference, the png library is required. Please define BUDDY_ENABLE_PNG for CMake.")
endif()
set(BUDDY_LENET_LIBS LENET mlir_c_runner_utils mlir_async_runtime mlir_runner_utils ${PNG_LIBRARIES})

target_link_libraries(buddy-lenet-run-test-cpu ${BUDDY_LENET_LIBS})

target_compile_definitions(buddy-lenet-run-test-cpu PRIVATE
    LENET_EXAMPLE_PATH="${LENET_EXAMPLE_PATH}"
    LENET_EXAMPLE_BUILD_PATH="${LENET_EXAMPLE_BUILD_PATH}"
)
