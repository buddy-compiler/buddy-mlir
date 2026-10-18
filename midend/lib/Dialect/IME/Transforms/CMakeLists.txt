add_mlir_dialect_library(BuddyIMETransforms
  LegalizeForLLVMExport.cpp
  LegalizeA100ForLLVMExport.cpp

  LINK_LIBS PUBLIC
  BuddyIME
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
)
