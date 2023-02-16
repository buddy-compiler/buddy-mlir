add_mlir_library(LowerGemminiPass
  LowerGemminiPass.cpp

  LINK_LIBS PUBLIC
  BuddyGemminiTransforms
  )
