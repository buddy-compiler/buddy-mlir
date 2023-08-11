_add_test_executable(buddy-container-test
  ContainerTest.cpp
)

if(BUDDY_ENABLE_OPENCV)
find_package(OpenCV REQUIRED CONFIG)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(JPEG REQUIRED)
include_directories(${JPEG_INCLUDE_DIR})
_add_test_executable(buddy-image-container-test
  ImageContainerTest.cpp
  LINK_LIBS
    ${OpenCV_LIBS}
    ${JPEG_LIBRARIES}
)
endif()

_add_test_executable(buddy-audio-container-test
  AudioContainerTest.cpp
)
