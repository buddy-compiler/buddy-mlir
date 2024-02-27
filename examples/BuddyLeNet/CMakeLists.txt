add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/lenet.mlir ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/arg0.data
  COMMAND python3 ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/buddy-lenet-import.py
  COMMENT "Generating lenet.mlir and parameter files"
)

add_custom_command(
  OUTPUT lenet.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/lenet.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt 
            -eliminate-empty-tensors
            -convert-tensor-to-linalg 
            -linalg-bufferize
            -convert-linalg-to-affine-loops
            -lower-affine
            -func-bufferize
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
          ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${BUDDY_BINARY_DIR}/../examples/BuddyLeNet/lenet.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyLeNet/lenet.mlir
  COMMENT "Building lenet.o"
  VERBATIM)

add_library(LENET STATIC lenet.o)

SET_TARGET_PROPERTIES(LENET PROPERTIES LINKER_LANGUAGE C)

add_executable(buddy-lenet-run buddy-lenet-main.cpp)
target_link_directories(buddy-lenet-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR})

set(BUDDY_LENET_LIBS LENET mlir_c_runner_utils ${OpenCV_LIBS})
target_link_libraries(buddy-lenet-run ${BUDDY_LENET_LIBS})
