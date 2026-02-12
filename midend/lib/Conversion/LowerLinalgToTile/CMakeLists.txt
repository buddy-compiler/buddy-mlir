add_mlir_library(LowerLinalgToTilePass
  LowerLinalgToTile.cpp
  
  LINK_LIBS PUBLIC
  BuddyTile
  )

