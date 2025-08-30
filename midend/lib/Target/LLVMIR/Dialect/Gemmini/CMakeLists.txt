add_mlir_translation_library(BuddyGemminiToLLVMIRTranslation
  GemminiToLLVMIRTranslation.cpp

  DEPENDS
  BuddyGemminiConversionsIncGen
  buddy_intrinsics_gen
  
  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  BuddyGemmini
  )
