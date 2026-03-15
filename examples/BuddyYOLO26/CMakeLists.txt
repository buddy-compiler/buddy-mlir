set(YOLO26N_IMPORT_PYTHONPATH
  "${LLVM_BINARY_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_PYTHON_PACKAGES_DIR}:$ENV{PYTHONPATH}"
)

set(YOLO26N_IMPORT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/import-yolo26n.py)
if(TARGET BuddyMLIRPythonModules)
  list(APPEND YOLO26N_IMPORT_DEPENDS BuddyMLIRPythonModules)
endif()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
         ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
  COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${YOLO26N_IMPORT_PYTHONPATH}"
          ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-yolo26n.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${YOLO26N_IMPORT_DEPENDS}
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files for YOLO26n"
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
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM
)


add_custom_command(
  OUTPUT subgraph0.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
            -pass-pipeline
            "builtin.module(func.func(tosa-to-linalg-named, tosa-to-arith, tosa-to-linalg, tosa-to-tensor))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -convert-elementwise-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
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
  VERBATIM
)

add_library(YOLO26N STATIC subgraph0.o forward.o)
set_target_properties(YOLO26N PROPERTIES LINKER_LANGUAGE C)

add_executable(buddy-yolo26n-run buddy-yolo26n-main.cpp)

set(YOLO26_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(YOLO26_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(buddy-yolo26n-run PRIVATE
  YOLO26_EXAMPLE_PATH="${YOLO26_EXAMPLE_PATH}"
  YOLO26_EXAMPLE_BUILD_PATH="${YOLO26_EXAMPLE_BUILD_PATH}"
)

target_link_directories(buddy-yolo26n-run PRIVATE ${LLVM_LIBRARY_DIR})

set(BUDDY_YOLO26_LIBS YOLO26N mlir_c_runner_utils BuddyLibDIP ${PNG_LIBRARIES})
target_link_libraries(buddy-yolo26n-run ${BUDDY_YOLO26_LIBS})
