add_library(Utils Utils.cpp)

add_library(DIPUtils DIPUtils.cpp)

add_library(WarpAffineUtils WarpAffineUtils.cpp)
target_link_libraries(DIPUtils
  PRIVATE
  Utils
  )
