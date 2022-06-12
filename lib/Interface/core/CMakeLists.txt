add_library(Container Container.cpp)
add_library(Mat Mat.cpp)

if(BUDDY_ENABLE_OPENCV)
  find_package(OpenCV REQUIRED CONFIG)
  include_directories(${OpenCV_INCLUDE_DIRS})
  add_library(ImageContainer ImageContainer.cpp)
endif()
