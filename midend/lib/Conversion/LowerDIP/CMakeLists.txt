add_mlir_library(LowerDIPPass
  LowerDIPPass.cpp

  LINK_LIBS PUBLIC
  BuddyDIPUtils
  )
