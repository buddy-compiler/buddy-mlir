add_mlir_translation_library(BuddyRVVToLLVMIRTranslation
  RVVToLLVMIRTranslation.cpp

  DEPENDS
  BuddyRVVConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  BuddyRVV
  )
