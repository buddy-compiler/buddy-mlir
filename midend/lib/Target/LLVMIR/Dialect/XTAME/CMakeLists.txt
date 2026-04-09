add_mlir_translation_library(MLIRXTAMEToLLVMIRTranslation
  XTAMEToLLVMIRTranslation.cpp

  DEPENDS
  BuddyXTAMEConversionsIncGen
  buddy_intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  BuddyXTAME
  )
