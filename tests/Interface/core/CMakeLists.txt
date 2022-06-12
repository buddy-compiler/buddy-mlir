_add_test_executable(buddy-container-test
  ContainerTest.cpp
  LINK_LIBS
    Container
)

_add_test_executable(buddy-mat-test
  MatTest.cpp
  LINK_LIBS
    Mat)

if(BUDDY_ENABLE_OPENCV)
find_package(OpenCV REQUIRED CONFIG)
include_directories(${OpenCV_INCLUDE_DIRS})
_add_test_executable(buddy-image-container-test
  ImageContainerTest.cpp
  LINK_LIBS
    ${OpenCV_LIBS}
)
endif()
