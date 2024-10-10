_add_test_executable(buddy-container-test
  ContainerTest.cpp
)


if (BUDDY_MLIR_ENABLE_DIP_LIB)
  set(NEW_DIP_LIBS ${PNG_LIBRARIES})
  _add_test_executable(buddy-new-image-container-test
    NewImageContainerTest.cpp
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
