add_mlir_dialect_library(BuddyIMETransforms
  LegalizeForLLVMExport.cpp
  LegalizeA100ForLLVMExport.cpp
  LegalizeFP16Kernels.cpp

  LINK_LIBS PUBLIC
  BuddyIME
  MLIRIR
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
)
