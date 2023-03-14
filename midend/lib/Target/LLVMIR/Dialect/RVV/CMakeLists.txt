add_mlir_translation_library(BuddyRVVToLLVMIRTranslation
  RVVToLLVMIRTranslation.cpp

  DEPENDS
  BuddyRVVConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBuddyLLVMDialect
  MLIRSupport
  MLIRBuddyTargetLLVMIRExport
  BuddyRVV
  )
