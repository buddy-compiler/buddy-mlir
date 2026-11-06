add_mlir_library(BuddyBOSCAMETransforms
  FPGAAMETarget.cpp
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
  MLIRSCFTransforms
  MLIRTransforms
)
