if (${BUDDY_DAP_OPT_VECTOR_SPLITTING})
  set(SPLITING_SIZE ${BUDDY_DAP_OPT_VECTOR_SPLITTING})
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

#-------------------------------------------------------------------------------
# Buddy DAP Dialect FIR operation
#-------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------
# Buddy DAP Dialect Biquad Operation
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT buddy-biquad.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt
  ${BUDDY_EXAMPLES_DIR}/DAPDialect/biquad.mlir
  -lower-dap="DAP-vector-splitting=${SPLITING_SIZE}" -convert-linalg-to-affine-loops
  -lower-affine
  -convert-scf-to-cf -convert-vector-to-llvm 
  --llvm-request-c-wrappers
  -convert-arith-to-llvm
  -convert-memref-to-llvm -convert-func-to-llvm
  -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
  ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR} 
  --filetype=obj -o ${BUDDY_BINARY_DIR}/../examples/DAPDialect/buddy-biquad.o
DEPENDS buddy-opt)

add_library(BuddyBiquad STATIC buddy-biquad.o)
set_target_properties(BuddyBiquad PROPERTIES LINKER_LANGUAGE CXX)

add_executable(biquad biquad.cpp)
add_dependencies(biquad buddy-opt)
target_link_libraries(biquad BuddyBiquad)

#-------------------------------------------------------------------------------
# Buddy DAP Dialect IIR Operation
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT buddy-iir.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt
  ${BUDDY_EXAMPLES_DIR}/DAPDialect/iir.mlir
  -lower-dap="DAP-vector-splitting=${SPLITING_SIZE}" -convert-linalg-to-affine-loops
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
