add_mlir_library(BuddyAMETransforms
  LegalizeForLLVMExport.cpp

  DEPENDS
  MLIRAMEIncGen

  LINK_LIBS PUBLIC
  BuddyAME
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
)
