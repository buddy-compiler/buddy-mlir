add_mlir_library(BuddyXTAMETransforms
  LegalizeForLLVMExport.cpp

  DEPENDS
  MLIRXTAMEIncGen

  LINK_LIBS PUBLIC
  BuddyXTAME
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRTransforms
)
