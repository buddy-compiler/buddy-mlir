add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyF16Test/f16test.mlir
  COMMAND ${Python3_EXECUTABLE} ${BUDDY_EXAMPLES_DIR}/BuddyF16Test/import-f16test.py
  COMMENT "Generating f16test.mlir..."
)

set(BUDDY_OPT_OPTIONS -one-shot-bufferize -convert-linalg-to-affine-loops -affine-loop-fusion -lower-affine -func-bufferize -arith-bufferize -tensor-bufferize -buffer-deallocation -finalizing-bufferize   -arith-expand -convert-arith-to-llvm -finalize-memref-to-llvm -convert-scf-to-cf -llvm-request-c-wrappers -convert-arith-to-llvm -convert-math-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts
)

add_custom_command(
  OUTPUT ${BUDDY_BINARY_DIR}/../examples/BuddyF16Test/f16test.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyF16Test/f16test.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          # tee ${BUDDY_EXAMPLES_DIR}/BuddyF16Test/mlir-opt-ret-f16test.mlir |   # for debug
          ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_OPT_OPTIONS} |
          # tee ${BUDDY_EXAMPLES_DIR}/BuddyF16Test/buddy-opt-ret-f16test.mlir |  # for debug
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          # tee ${BUDDY_EXAMPLES_DIR}/BuddyF16Test/f16test.llvm |                # for debug
          ${LLVM_MLIR_BINARY_DIR}/llvm-as |
          ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -mattr=-sse2  # prevent float ABI bug on x86-64
            -o ${BUDDY_BINARY_DIR}/../examples/BuddyF16Test/f16test.o
  DEPENDS ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_EXAMPLES_DIR}/BuddyF16Test/f16test.mlir
  COMMENT "Building f16test.o "
  VERBATIM
)

add_library(F16TEST STATIC f16test.o)

SET_TARGET_PROPERTIES(
  F16TEST
  PROPERTIES
  LINKER_LANGUAGE C
)

add_executable(buddy-f16test-run f16test-main.cpp fp16_h2f.cpp fp16_f2h.cpp bp16_f2b.cpp)
target_include_directories(buddy-f16test-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR}/../../../)

set(BUDDY_F16TEST_LIBS
  F16TEST
  mlir_c_runner_utils
)

target_link_libraries(buddy-f16test-run ${BUDDY_F16TEST_LIBS})