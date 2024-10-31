_add_test_executable(buddy-container-test
  ContainerTest.cpp
)

if(BUDDY_ENABLE_OPENCV)
  find_package(OpenCV REQUIRED CONFIG)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

if(BUDDY_MLIR_ENABLE_DIP_LIB AND BUDDY_ENABLE_OPENCV)
  set(DIP_LIBS ${JPEG_LIBRARY} ${PNG_LIBRARY})
  _add_test_executable(buddy-image-container-test
    ImageContainerTest.cpp
    LINK_LIBS
      ${OpenCV_LIBS}
      ${DIP_LIBS}
  )
endif()

if(BUDDY_MLIR_ENABLE_DIP_LIB)
  set(NEW_DIP_LIBS "")
  if(BUDDY_ENABLE_PNG)
    list(APPEND NEW_DIP_LIBS ${PNG_LIBRARIES})
    _add_test_executable(buddy-new-image-container-test-png
      NewImageContainerTestPng.cpp
      LINK_LIBS
        ${NEW_DIP_LIBS}
    )
  endif()
  _add_test_executable(buddy-new-image-container-test-bmp
    NewImageContainerTestBmp.cpp
    LINK_LIBS
      ${NEW_DIP_LIBS}
  )
endif()

_add_test_executable(buddy-audio-container-test
  AudioContainerTest.cpp
)

_add_test_executable(buddy-text-container-test
  TextContainerTest.cpp
)
