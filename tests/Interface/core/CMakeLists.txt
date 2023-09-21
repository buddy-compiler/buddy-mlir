_add_test_executable(buddy-container-test
  ContainerTest.cpp
)

if(BUDDY_ENABLE_OPENCV)
find_package(OpenCV REQUIRED CONFIG)
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${JPEG_INCLUDE_DIRS})
include_directories(${PNG_INCLUDE_DIRS})
_add_test_executable(buddy-image-container-test
  ImageContainerTest.cpp
  LINK_LIBS
    ${OpenCV_LIBS}
    ${JPEG_LIBRARY}
    ${PNG_LIBRARY}
)
endif()

_add_test_executable(buddy-audio-container-test
  AudioContainerTest.cpp
)

_add_test_executable(buddy-text-container-test
  TextContainerTest.cpp
)
