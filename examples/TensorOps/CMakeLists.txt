add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/TensorOps/forward.mlir
  COMMAND python3 ${BUDDY_EXAMPLES_DIR}/TensorOps/tensor-ops-import.py
  COMMENT "Building forward.mlir"
)

add_custom_command(
  OUTPUT forward.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/TensorOps/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt 
            -pass-pipeline "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), eliminate-empty-tensors, func.func(llvm-request-c-wrappers),convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf,  convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llvm-as |
          ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${BUDDY_BINARY_DIR}/../examples/TensorOps/forward.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/TensorOps/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_library(TENSOROPS STATIC forward.o)

SET_TARGET_PROPERTIES(TENSOROPS PROPERTIES LINKER_LANGUAGE C)

add_executable(tensor-ops-run main.cpp)
target_link_directories(tensor-ops-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR})

set(TENSOR_OPS_LIBS TENSOROPS mlir_c_runner_utils)
target_link_libraries(tensor-ops-run ${TENSOR_OPS_LIBS})
