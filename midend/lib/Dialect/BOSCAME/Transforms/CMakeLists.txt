add_mlir_library(BuddyBOSCAMETransforms
  LegalizeForLLVMExport.cpp

  DEPENDS
  MLIRBOSCAMEIncGen

  LINK_LIBS PUBLIC
  BuddyBOSCAME
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRTransforms
)
