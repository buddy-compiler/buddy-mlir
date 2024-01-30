add_mlir_library(LowerGemminiPass
  LowerGemminiPass.cpp
  DEPENDS
  
  PassesIncGen
  LINK_LIBS PUBLIC
  BuddyGemminiTransforms
  )
