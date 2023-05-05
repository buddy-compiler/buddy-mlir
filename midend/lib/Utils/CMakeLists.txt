add_library(Utils Utils.cpp)

add_library(DIPUtils DIPUtils.cpp)

add_library(AffineTransformUtils AffineTransformUtils.cpp)
target_link_libraries(DIPUtils
  PRIVATE
  Utils
  AffineTransformUtils
  )
