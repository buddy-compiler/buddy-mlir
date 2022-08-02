_add_test_executable(buddy-container-test
  ContainerTest.cpp
  LINK_LIBS
    Container
)

if(BUDDY_ENABLE_OPENCV)
find_package(OpenCV REQUIRED CONFIG)
include_directories(${OpenCV_INCLUDE_DIRS})
_add_test_executable(buddy-image-container-test
  ImageContainerTest.cpp
  LINK_LIBS
    ${OpenCV_LIBS}
)
endif()

_add_test_executable(buddy-audio-container-test
  AudioContainerTest.cpp
  LINK_LIBS
  Container AudioContainer
)
