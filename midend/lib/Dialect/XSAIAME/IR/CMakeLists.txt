add_mlir_dialect_library(BuddyXSAIAME
  XSAIAMEDialect.cpp
  XSAIAMETypes.cpp

  DEPENDS
  MLIRXSAIAMEIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBytecodeOpInterface
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
