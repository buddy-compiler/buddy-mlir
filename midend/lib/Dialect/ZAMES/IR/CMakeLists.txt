add_mlir_dialect_library(BuddyZAMES
  ZAMESDialect.cpp
  ZAMESTypes.cpp

  DEPENDS
  MLIRZAMESIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBytecodeOpInterface
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
