add_mlir_library(BuddyUtils
  Utils.cpp
  DIPUtils.cpp
  DAPUtils.cpp
  AffineTransformUtils.cpp
  )

add_mlir_library(BuddyDIPUtils
  DIPUtils.cpp
  
  LINK_LIBS PUBLIC
  BuddyUtils
  )

add_mlir_library(BuddyDAPUtils
  DAPUtils.cpp
  
  LINK_LIBS PUBLIC
  BuddyUtils
  )
