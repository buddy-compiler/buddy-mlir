add_custom_command(OUTPUT fir.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt 
  ${BUDDY_EXAMPLES_DIR}/DAPDialect/fir.mlir
  -lower-dap --convert-linalg-to-affine-loops -lower-affine
  -convert-scf-to-cf -convert-vector-to-llvm -convert-memref-to-llvm
  -llvm-request-c-wrappers -convert-func-to-llvm -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
  ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_TARGET_TRIPLE} -mattr=${BUDDY_OPT_ATTR}
  --filetype=obj -o ${BUDDY_BINARY_DIR}/../examples/DAPDialect/fir.o
DEPENDS buddy-opt)

add_library(FIR STATIC fir.o)

SET_TARGET_PROPERTIES(
  FIR
  PROPERTIES
  LINKER_LANGUAGE CXX)

add_executable(FirLowpass FirLowpass.cpp)
add_dependencies(FirLowpass buddy-opt)
target_link_libraries(FirLowpass FIR)
