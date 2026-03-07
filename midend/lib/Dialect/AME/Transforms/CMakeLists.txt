add_mlir_library(BuddyAMETransforms
  LegalizeForLLVMExport.cpp

  DEPENDS
  MLIRAMEIncGen

  LINK_LIBS PUBLIC
  BuddyAME
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRTransforms
)
