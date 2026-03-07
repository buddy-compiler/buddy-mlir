add_mlir_dialect_library(BuddyBuckyballTransforms
  LegalizeForLLVMExport.cpp

  LINK_LIBS PUBLIC
  BuddyBuckyball
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
