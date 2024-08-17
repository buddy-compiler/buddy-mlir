add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyResNet18/arg0.data
         ${BUDDY_EXAMPLES_DIR}/BuddyResNet18/arg1.data
         ${BUDDY_EXAMPLES_DIR}/BuddyResNet18/forward.mlir
         ${BUDDY_EXAMPLES_DIR}/BuddyResNet18/subgraph0.mlir
  COMMAND python3 ${BUDDY_EXAMPLES_DIR}/BuddyResNet18/import-resnet18.py
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)


# Command to build the MLIR object file
add_custom_command(
  OUTPUT forward.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyResNet18/forward.mlir
            -pass-pipeline
            "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), \
            empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, \
            func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -pass-pipeline
            "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), \
            eliminate-empty-tensors, func.func(llvm-request-c-wrappers), \
            convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf, \
            convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, \
            convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llvm-as |
          ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3 
            -o ${BUDDY_BINARY_DIR}/../examples/BuddyResNet18/forward.o
            DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyResNet18/forward.mlir
            COMMENT "Building forward.o"
            VERBATIM)

add_custom_command(
  OUTPUT subgraph0.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_EXAMPLES_DIR}/BuddyResNet18/subgraph0.mlir 
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
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llvm-as |
          ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3 
            -o ${BUDDY_BINARY_DIR}/../examples/BuddyResNet18/subgraph0.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyResNet18/subgraph0.mlir
          buddy-opt
  COMMENT "Building subgraph0.o"
  VERBATIM)

# Create the library from the object file
add_library(RESNET18 STATIC subgraph0.o forward.o)

SET_TARGET_PROPERTIES(RESNET18 PROPERTIES LINKER_LANGUAGE C)

# Add the executable
add_executable(resnet18-run resnet18-main.cpp)
target_link_directories(resnet18-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR})

# Link the required libraries
set(BUDDY_RESNET18_LIBS RESNET18 mlir_c_runner_utils ${OpenCV_LIBS})
target_link_libraries(resnet18-run ${BUDDY_RESNET18_LIBS})