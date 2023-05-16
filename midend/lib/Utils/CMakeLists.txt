add_mlir_library(BuddyUtils
  Utils.cpp
  DIPUtils.cpp
  )

add_mlir_library(BuddyDIPUtils
  DIPUtils.cpp
  
  LINK_LIBS PUBLIC
  BuddyUtils
  )

add_library(Utils Utils.cpp)

add_library(DIPUtils DIPUtils.cpp)
target_link_libraries(DIPUtils
  PRIVATE
  Utils
  )