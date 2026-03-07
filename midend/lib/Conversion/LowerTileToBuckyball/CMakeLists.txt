add_mlir_library(LowerTileToBuckyballPass
  LowerTileToBuckyball.cpp
  
  LINK_LIBS PUBLIC
  BuddyTile
  BuddyBuckyball
  )

