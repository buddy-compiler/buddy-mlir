# set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/Dialect/MQ/MQHigh/MQHighOps.td)

# # Set include directory for tablegen
# set(MLIR_TABLEGEN_INCLUDE_DIRS
#   ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/Dialect/MQ/MQHigh
#   ${MLIR_TABLEGEN_INCLUDE_DIRS}
# )

# mlir_tablegen(MQHighOps.h.inc -gen-op-decls)
# mlir_tablegen(MQHighOps.cpp.inc -gen-op-defs)
# add_public_tablegen_target(MQHighOpsIncGen)

add_mlir_dialect_library(MQHigh
  MQHighDialect.cpp
  MQHighOps.cpp
  MQHighOps/MatMul.cpp

  # DEPENDS
  # MQHighOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
#   MLIRSideEffectInterfaces
  MLIRInferTypeOpInterface
)
