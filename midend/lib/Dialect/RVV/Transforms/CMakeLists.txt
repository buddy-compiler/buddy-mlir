add_mlir_dialect_library(BuddyRVVTransforms
  LegalizeForLLVMExport.cpp
  Arithmetic.cpp

  DEPENDS
  BuddyRVVConversionsIncGen

  LINK_LIBS PUBLIC
  BuddyRVV
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
