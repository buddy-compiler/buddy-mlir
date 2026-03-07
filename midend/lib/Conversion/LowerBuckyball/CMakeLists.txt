add_mlir_library(LowerBuckyballPass
  LowerBuckyballPass.cpp

  LINK_LIBS PUBLIC
  BuddyBuckyballTransforms
  )
