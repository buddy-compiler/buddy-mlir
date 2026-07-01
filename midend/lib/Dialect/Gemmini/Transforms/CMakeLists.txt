add_mlir_dialect_library(BuddyGemminiTransforms
  LegalizeForLLVMExport.cpp
  LegalizeForFuncExport.cpp

  LINK_LIBS PUBLIC
  BuddyGemmini
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
