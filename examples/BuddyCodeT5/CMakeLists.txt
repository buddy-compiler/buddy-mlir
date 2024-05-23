add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyCodeT5/forward.mlir ${BUDDY_EXAMPLES_DIR}/BuddyCodeT5/subgraph0.mlir ${BUDDY_EXAMPLES_DIR}/BuddyCodeT5/arg0.data ${BUDDY_EXAMPLES_DIR}/BuddyCodeT5/arg1.data
  COMMAND python3 ${BUDDY_EXAMPLES_DIR}/BuddyCodeT5/import-bert.py
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)


add_custom_command(
  OUTPUT forward.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyCodeT5/forward.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt 
            -pass-pipeline "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), eliminate-empty-tensors, func.func(llvm-request-c-wrappers),convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf,  convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llvm-as |
          ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${BUDDY_BINARY_DIR}/../examples/BuddyCodeT5/forward.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyCodeT5/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_EXAMPLES_DIR}/BuddyCodeT5/subgraph0.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, func-bufferize-dynamic-offset, arith-bufferize, func.func(linalg-bufferize, tensor-bufferize))" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt 
            -pass-pipeline "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), eliminate-empty-tensors, func.func(llvm-request-c-wrappers),convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf,  convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llvm-as |
          ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${BUDDY_BINARY_DIR}/../examples/BuddyCodeT5/subgraph0.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyCodeT5/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM)

add_library(CodeT5 STATIC forward.o subgraph0.o)

SET_TARGET_PROPERTIES(CodeT5 PROPERTIES LINKER_LANGUAGE C)

add_executable(buddy-codet5-run codet5-main.cpp)
target_link_directories(buddy-codet5-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR})

set(BUDDY_CODET5_LIBS CodeT5 mlir_c_runner_utils)
target_link_libraries(buddy-codet5-run ${BUDDY_BERT_LIBS})
