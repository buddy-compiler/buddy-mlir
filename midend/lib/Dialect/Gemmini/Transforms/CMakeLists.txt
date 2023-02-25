add_mlir_dialect_library(BuddyGemminiTransforms
  LegalizeForLLVMExport.cpp

  LINK_LIBS PUBLIC
  BuddyGemmini
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
