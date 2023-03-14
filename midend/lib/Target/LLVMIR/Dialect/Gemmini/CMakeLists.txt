add_mlir_translation_library(BuddyGemminiToLLVMIRTranslation
  GemminiToLLVMIRTranslation.cpp

  DEPENDS
  BuddyGemminiConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBuddyLLVMDialect
  MLIRSupport
  MLIRBuddyTargetLLVMIRExport
  BuddyGemmini
  )
