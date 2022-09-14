if (${BUDDY_DIP_OPT_STRIP_MINING})
  set(SPLITING_SIZE ${BUDDY_DIP_OPT_STRIP_MINING})
elseif(HAVE_AVX512)
  set(SPLITING_SIZE 256)
elseif(HAVE_AVX2)
  set(SPLITING_SIZE 128)
elseif(HAVE_SSE)
  set(SPLITING_SIZE 64)
elseif(HAVE_NEON)
  set(SPLITING_SIZE 64)
endif()

message(STATUS "Spliting size: ${SPLITING_SIZE}")

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

add_executable(firLowpass firLowpass.cpp)
add_dependencies(firLowpass buddy-opt)
target_link_libraries(firLowpass FIR)


add_custom_command(OUTPUT buddy-iir.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt
  ${BUDDY_EXAMPLES_DIR}/DAPDialect/iir.mlir
  -lower-dap="DAP-strip-mining=${SPLITING_SIZE}" -convert-linalg-to-affine-loops
  -lower-affine -convert-scf-to-cf -convert-vector-to-llvm 
  --llvm-request-c-wrappers -convert-arith-to-llvm
  -convert-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
  ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR} 
  --filetype=obj -o ${BUDDY_BINARY_DIR}/../examples/DAPDialect/buddy-iir.o
DEPENDS buddy-opt)

add_library(IIR STATIC buddy-iir.o)

SET_TARGET_PROPERTIES(
  IIR
  PROPERTIES
  LINKER_LANGUAGE CXX)

  add_executable(iirLowpass iirLowpass.cpp)
  add_dependencies(iirLowpass buddy-opt)
  target_link_libraries(iirLowpass IIR)
  