add_mlir_dialect_library(BuddyRVVTransforms
  LegalizeForLLVMExport.cpp

  DEPENDS
  BuddyRVVConversionsIncGen

  LINK_LIBS PUBLIC
  BuddyRVV
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
