add_library(Utils Utils.cpp)

add_library(DIPUtils DIPUtils.cpp)
target_link_libraries(DIPUtils
  PRIVATE
  Utils
  )
