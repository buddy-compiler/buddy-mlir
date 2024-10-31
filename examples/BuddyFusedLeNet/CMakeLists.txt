file(GLOB SUBGRAPH_FILES "${CMAKE_CURRENT_SOURCE_DIR}/subgraph*.mlir")

add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyFusedLeNet/arg0.data
         ${BUDDY_EXAMPLES_DIR}/BuddyFusedLeNet/forward.mlir
         ${SUBGRAPH_FILES}
  COMMAND python3 ${BUDDY_EXAMPLES_DIR}/BuddyFusedLeNet/buddy-lenet-import.py
  COMMENT "Generating forward.mlir, subgraph*.mlir and parameter files"
)

set(SUBGRAPH_OBJECTS "")
foreach(SUBGRAPH_FILE ${SUBGRAPH_FILES})
  get_filename_component(SUBGRAPH_NAME ${SUBGRAPH_FILE} NAME_WE)
  set(OBJECT_FILE ${BUDDY_BINARY_DIR}/../examples/BuddyFusedLeNet/${SUBGRAPH_NAME}.o)
  list(APPEND SUBGRAPH_OBJECTS ${OBJECT_FILE})

  add_custom_command(
    OUTPUT ${OBJECT_FILE}
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${SUBGRAPH_FILE}
              -pass-pipeline
              "builtin.module(func.func(tosa-to-linalg-named, tosa-to-arith, tosa-to-linalg, tosa-to-tensor))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -convert-elementwise-to-linalg
              -func-bufferize-dynamic-offset
              -arith-bufferize
              -func-bufferize
              -tensor-bufferize
              -linalg-bufferize
              -finalizing-bufferize
              -convert-linalg-to-loops
              -lower-affine
              -convert-scf-to-cf
              -llvm-request-c-wrappers
              -convert-math-to-llvm
              -convert-math-to-libm
              -convert-arith-to-llvm
              -convert-func-to-llvm
              -expand-strided-metadata
              -finalize-memref-to-llvm
              -reconcile-unrealized-casts |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
            ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
            ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
              -o ${OBJECT_FILE}
    DEPENDS ${SUBGRAPH_FILE} buddy-opt
    COMMENT "Building ${SUBGRAPH_NAME}.o"
    VERBATIM
  )
endforeach()

add_custom_command(
  OUTPUT forward.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyFusedLeNet/forward.mlir
            -pass-pipeline
            "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), \
            empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, \
            func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline
            "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), \
            eliminate-empty-tensors, func.func(llvm-request-c-wrappers), \
            convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf, \
            convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, \
            convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${BUDDY_BINARY_DIR}/../examples/BuddyFusedLeNet/forward.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyFusedLeNet/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM
)

add_library(FUSED_LENET STATIC ${SUBGRAPH_OBJECTS} forward.o)

SET_TARGET_PROPERTIES(FUSED_LENET PROPERTIES LINKER_LANGUAGE C)

add_executable(buddy-fused-lenet-run buddy-lenet-main.cpp)
target_link_directories(buddy-fused-lenet-run PRIVATE ${LLVM_LIBRARY_DIR})

set(BUDDY_LENET_LIBS FUSED_LENET mlir_c_runner_utils ${PNG_LIBRARIES})

target_link_libraries(buddy-fused-lenet-run ${BUDDY_LENET_LIBS})
