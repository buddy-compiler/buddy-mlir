add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/forward.mlir ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/subgraph0.mlir ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/arg0.data
  COMMAND python3 ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/buddy-lenet-import.py
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)

add_custom_command(
  OUTPUT forward.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/forward.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt 
            -pass-pipeline "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), eliminate-empty-tensors, func.func(llvm-request-c-wrappers),convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf,  convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llvm-as |
          ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/forward.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/subgraph0.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-tensor-to-linalg 
            -linalg-bufferize
            -convert-linalg-to-affine-loops
            -lower-affine
            -func-bufferize-dynamic-offset
            -arith-bufferize
            -tensor-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -convert-vector-to-scf
            -expand-strided-metadata
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llvm-as |
          ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/subgraph0.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM)

add_library(LENET STATIC subgraph0.o forward.o)

SET_TARGET_PROPERTIES(LENET PROPERTIES LINKER_LANGUAGE C)

add_executable(buddy-lenet-run buddy-lenet-main.cpp)
target_link_directories(buddy-lenet-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR})

set(BUDDY_LENET_LIBS LENET mlir_c_runner_utils ${OpenCV_LIBS})
target_link_libraries(buddy-lenet-run ${BUDDY_LENET_LIBS})
