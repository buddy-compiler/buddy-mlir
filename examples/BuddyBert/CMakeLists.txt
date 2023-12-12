add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyBert/bert.mlir ${BUDDY_EXAMPLES_DIR}/BuddyBert/arg0.data ${BUDDY_EXAMPLES_DIR}/BuddyBert/arg1.data
  COMMAND python3 ${BUDDY_EXAMPLES_DIR}/BuddyBert/import-bert.py
  COMMENT "Generating bert.mlir and parameter files"
)


add_custom_command(
  OUTPUT bert.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyBert/bert.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt 
            -pass-pipeline "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), eliminate-empty-tensors, func.func(llvm-request-c-wrappers),convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf,  convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llvm-as |
          ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${BUDDY_BINARY_DIR}/../examples/BuddyBert/bert.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyBert/bert.mlir
  COMMENT "Building bert.o"
  VERBATIM)

add_library(BERT STATIC bert.o)

SET_TARGET_PROPERTIES(BERT PROPERTIES LINKER_LANGUAGE C)

add_executable(buddy-bert-run bert-main.cpp)
target_link_directories(buddy-bert-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR})

set(BUDDY_BERT_LIBS BERT mlir_c_runner_utils)
target_link_libraries(buddy-bert-run ${BUDDY_BERT_LIBS})
