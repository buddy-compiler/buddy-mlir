add_mlir_translation_library(MLIRAMEToLLVMIRTranslation
  AMEToLLVMIRTranslation.cpp

  DEPENDS
  BuddyAMEConversionsIncGen
  buddy_intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  BuddyAME
  )
  