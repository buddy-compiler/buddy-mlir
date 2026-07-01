add_mlir_library(LowerGemminiToFuncPass
  LowerGemminiToFunc.cpp

  LINK_LIBS PUBLIC
  BuddyGemminiTransforms
  )
