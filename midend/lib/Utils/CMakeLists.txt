add_mlir_library(BuddyUtils
  Utils.cpp
  DIPUtils.cpp
  )

add_mlir_library(BuddyDIPUtils
  DIPUtils.cpp
  
  LINK_LIBS PUBLIC
  BuddyUtils
  )