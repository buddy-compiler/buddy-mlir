add_mlir_library(BuddyDynamicVLOpInterface
  DynamicVLOpInterface.cpp

  DEPENDS
  MLIRDynamicVLOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
