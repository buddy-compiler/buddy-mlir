add_mlir_translation_library(MLIRBOSCAMEToLLVMIRTranslation
  BOSCAMEToLLVMIRTranslation.cpp

  DEPENDS
  BuddyBOSCAMEConversionsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  BuddyBOSCAME
  )
