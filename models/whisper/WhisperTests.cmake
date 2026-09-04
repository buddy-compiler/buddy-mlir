set(_WHISPER_RUNTIME_TEST_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/runtime-test")
set(_WHISPER_RUNTIME_TEST_RAX
  "${_WHISPER_RUNTIME_TEST_DIR}/whisper-runtime-test.rax")

add_library(buddy-whisper-runtime-fake-kernel MODULE FakeWhisperKernel.cpp)
target_compile_features(buddy-whisper-runtime-fake-kernel PRIVATE cxx_std_17)
target_include_directories(buddy-whisper-runtime-fake-kernel PRIVATE
  "${BUDDY_SOURCE_DIR}/frontend/Interfaces")
set_target_properties(buddy-whisper-runtime-fake-kernel PROPERTIES
  PREFIX ""
  OUTPUT_NAME "whisper-runtime-fake-kernel")

add_custom_command(
  OUTPUT "${_WHISPER_RUNTIME_TEST_RAX}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_WHISPER_RUNTIME_TEST_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy
          "$<TARGET_FILE:buddy-whisper-runtime-fake-kernel>"
          "${_WHISPER_RUNTIME_TEST_DIR}/model.so"
  COMMAND ${CMAKE_COMMAND} -E copy
          "$<TARGET_FILE:buddy_models_whisper_runner>"
          "${_WHISPER_RUNTIME_TEST_DIR}/whisper_runner.so"
  COMMAND ${CMAKE_COMMAND} -E copy
          "${CMAKE_CURRENT_SOURCE_DIR}/WhisperRuntimeTestVocab.txt"
          "${_WHISPER_RUNTIME_TEST_DIR}/vocab.txt"
  COMMAND ${CMAKE_COMMAND} -E copy
          "${CMAKE_CURRENT_SOURCE_DIR}/WhisperRuntimeTestWeights.bin"
          "${_WHISPER_RUNTIME_TEST_DIR}/weights.bin"
  COMMAND ${CMAKE_COMMAND} -E copy
          "${CMAKE_SOURCE_DIR}/examples/BuddyWhisper/audio.wav"
          "${_WHISPER_RUNTIME_TEST_DIR}/audio.wav"
  COMMAND $<TARGET_FILE:rax-pack>
          "${CMAKE_CURRENT_SOURCE_DIR}/WhisperRuntimeTest.mlir"
          -o "${_WHISPER_RUNTIME_TEST_RAX}"
  DEPENDS
    rax-pack
    buddy_models_whisper_runner
    buddy-whisper-runtime-fake-kernel
    "${CMAKE_CURRENT_SOURCE_DIR}/WhisperRuntimeTest.mlir"
    "${CMAKE_CURRENT_SOURCE_DIR}/WhisperRuntimeTestVocab.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/WhisperRuntimeTestShortVocab.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/WhisperRuntimeTestWeights.bin"
    "${CMAKE_SOURCE_DIR}/examples/BuddyWhisper/audio.wav"
  VERBATIM)
add_custom_target(buddy-whisper-runtime-test-fixture
  DEPENDS "${_WHISPER_RUNTIME_TEST_RAX}")

add_executable(buddy-whisper-runtime-test WhisperRuntimeTest.cpp)
target_compile_features(buddy-whisper-runtime-test PRIVATE cxx_std_17)
target_link_libraries(buddy-whisper-runtime-test PRIVATE buddy_models_whisper)
add_dependencies(buddy-whisper-runtime-test
  buddy-whisper-runtime-test-fixture)

add_test(NAME whisper-runtime COMMAND buddy-whisper-runtime-test
  "${_WHISPER_RUNTIME_TEST_RAX}"
  "${CMAKE_SOURCE_DIR}/examples/BuddyWhisper/audio.wav"
  "${CMAKE_CURRENT_SOURCE_DIR}/WhisperRuntimeTestShortVocab.txt")
set_tests_properties(whisper-runtime PROPERTIES
  PASS_REGULAR_EXPRESSION "WhisperRuntime tests passed")
