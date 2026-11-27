add_mlir_dialect_library(BuddyXSAIAMETransforms
  LegalizeForLLVMExport.cpp

  LINK_LIBS PUBLIC
  BuddyXSAIAME
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransforms
)
