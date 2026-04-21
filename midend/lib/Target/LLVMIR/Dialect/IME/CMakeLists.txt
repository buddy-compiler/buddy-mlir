add_mlir_translation_library(MLIRIMEToLLVMIRTranslation
  IMEToLLVMIRTranslation.cpp

  DEPENDS
  BuddyIMEConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  BuddyIME
  )
