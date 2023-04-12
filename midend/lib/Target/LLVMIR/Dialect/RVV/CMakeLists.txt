add_mlir_translation_library(BuddyRVVToLLVMIRTranslation
  RVVToLLVMIRTranslation.cpp

  DEPENDS
  BuddyRVVConversionsIncGen
  buddy_intrinsics_gen
  
  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  BuddyRVV
  )
