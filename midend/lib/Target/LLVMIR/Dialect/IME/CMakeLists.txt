add_mlir_translation_library(MLIRIMEToLLVMIRTranslation
  IMEToLLVMIRTranslation.cpp

  DEPENDS
  BuddyIMEConversionsIncGen
  buddy_intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  BuddyIME
  )
