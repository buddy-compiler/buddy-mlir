add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyTest/forward.mlir
  COMMAND python3 ${BUDDY_EXAMPLES_DIR}/BuddyTest/import-test.py
  COMMENT "Generating forward.mlir"
)


add_custom_command(
  OUTPUT forward.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyTest/forward.mlir -gpu-kernel-outlining -llvm-request-c-wrappers -convert-vector-to-scf -convert-vector-to-llvm |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt 
            -pass-pipeline "builtin.module(nvvm-attach-target{chip=sm_75 O=3},  gpu.module(convert-scf-to-cf, convert-gpu-to-nvvm, convert-arith-to-llvm), convert-scf-to-cf, gpu-to-llvm, reconcile-unrealized-casts, gpu-module-to-binary)" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llvm-as |
          ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O3 -o ${BUDDY_BINARY_DIR}/../examples/BuddyTest/forward.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyTest/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)
  
  
add_library(TEST STATIC forward.o)

SET_TARGET_PROPERTIES(TEST PROPERTIES LINKER_LANGUAGE C)

add_executable(buddy-test-run test-main.cpp)
target_link_directories(buddy-test-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR})

set(BUDDY_TEST_LIBS TEST mlir_runner_utils mlir_cuda_runtime)
target_link_libraries(buddy-test-run ${BUDDY_TEST_LIBS})
