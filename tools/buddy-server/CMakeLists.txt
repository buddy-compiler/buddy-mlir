# tools/buddy-server/CMakeLists.txt

set(BUDDY_SERVER_COMMON_SOURCES
  JsonCodec.cpp
  AudioTranscriptionJsonCodec.cpp
  ResidentModelPluginHandle.cpp
  EmbeddingModelPluginHandle.cpp
  MaskedLMModelPluginHandle.cpp
  AudioTranscriptionModelPluginHandle.cpp
  BackendSelection.cpp
  SimpleHttpServer.cpp
)

add_executable(buddy-server
  buddy-server.cpp
  ${BUDDY_SERVER_COMMON_SOURCES}
)
target_compile_features(buddy-server PRIVATE cxx_std_17)
target_include_directories(buddy-server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(buddy-server PRIVATE
  buddy_runtime_core
  buddy_runtime_llm
  LLVMSupport
  ${CMAKE_DL_LIBS}
)
target_link_directories(buddy-server PRIVATE ${LLVM_LIBRARY_DIR})
if(BUDDY_MLIR_USE_MIMALLOC)
  target_link_libraries(buddy-server PRIVATE mimalloc)
endif()
install(TARGETS buddy-server
  EXPORT BuddyMLIRTargets
  COMPONENT buddy_tools)

if(BUDDY_ENABLE_TESTS)
  add_test(NAME buddy-server-help COMMAND buddy-server --help)
  set_tests_properties(buddy-server-help PROPERTIES
    PASS_REGULAR_EXPRESSION "Usage:.*buddy-server")

  add_executable(buddy-server-json-codec-test
    JsonCodecTest.cpp
    JsonCodec.cpp
    AudioTranscriptionJsonCodec.cpp)
  target_compile_features(buddy-server-json-codec-test PRIVATE cxx_std_17)
  target_include_directories(buddy-server-json-codec-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(buddy-server-json-codec-test PRIVATE
    buddy_runtime_core buddy_runtime_llm LLVMSupport ${CMAKE_DL_LIBS})
  target_link_directories(buddy-server-json-codec-test PRIVATE
    ${LLVM_LIBRARY_DIR})
  add_test(NAME buddy-server-json-codec
    COMMAND buddy-server-json-codec-test)
  set_tests_properties(buddy-server-json-codec PROPERTIES
    PASS_REGULAR_EXPRESSION "JsonCodec tests passed")

  add_executable(buddy-server-embedding-plugin-handle-test
    EmbeddingModelPluginHandleTest.cpp
    EmbeddingModelPluginHandle.cpp)
  target_compile_features(buddy-server-embedding-plugin-handle-test
    PRIVATE cxx_std_17)
  target_include_directories(buddy-server-embedding-plugin-handle-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(buddy-server-embedding-plugin-handle-test PRIVATE
    buddy_runtime_core ${CMAKE_DL_LIBS})
  add_test(NAME buddy-server-embedding-plugin-handle
    COMMAND buddy-server-embedding-plugin-handle-test)

  add_executable(buddy-server-masked-lm-plugin-handle-test
    MaskedLMModelPluginHandleTest.cpp
    MaskedLMModelPluginHandle.cpp)
  target_compile_features(buddy-server-masked-lm-plugin-handle-test
    PRIVATE cxx_std_17)
  target_include_directories(buddy-server-masked-lm-plugin-handle-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(buddy-server-masked-lm-plugin-handle-test PRIVATE
    buddy_runtime_core ${CMAKE_DL_LIBS})
  add_test(NAME buddy-server-masked-lm-plugin-handle
    COMMAND buddy-server-masked-lm-plugin-handle-test)

  add_library(buddy-server-transcription-fake-plugin MODULE
    AudioTranscriptionFakePlugin.cpp)
  add_library(buddy-server-transcription-missing-plugin MODULE
    AudioTranscriptionMissingSymbolPlugin.cpp)
  add_library(buddy-server-transcription-null-plugin MODULE
    AudioTranscriptionNullPlugin.cpp)
  foreach(_plugin
      buddy-server-transcription-fake-plugin
      buddy-server-transcription-missing-plugin
      buddy-server-transcription-null-plugin)
    target_compile_features(${_plugin} PRIVATE cxx_std_17)
    target_link_libraries(${_plugin} PRIVATE buddy_runtime_core)
  endforeach()

  add_executable(buddy-server-audio-transcription-plugin-handle-test
    AudioTranscriptionModelPluginHandleTest.cpp
    AudioTranscriptionModelPluginHandle.cpp)
  target_compile_features(
    buddy-server-audio-transcription-plugin-handle-test PRIVATE cxx_std_17)
  target_include_directories(
    buddy-server-audio-transcription-plugin-handle-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(
    buddy-server-audio-transcription-plugin-handle-test PRIVATE
    buddy_runtime_core ${CMAKE_DL_LIBS})
  target_compile_definitions(
    buddy-server-audio-transcription-plugin-handle-test PRIVATE
    BUDDY_TEST_TRANSCRIPTION_PLUGIN="$<TARGET_FILE:buddy-server-transcription-fake-plugin>"
    BUDDY_TEST_TRANSCRIPTION_MISSING_PLUGIN="$<TARGET_FILE:buddy-server-transcription-missing-plugin>"
    BUDDY_TEST_TRANSCRIPTION_NULL_PLUGIN="$<TARGET_FILE:buddy-server-transcription-null-plugin>")
  add_dependencies(buddy-server-audio-transcription-plugin-handle-test
    buddy-server-transcription-fake-plugin
    buddy-server-transcription-missing-plugin
    buddy-server-transcription-null-plugin)
  add_test(NAME buddy-server-audio-transcription-plugin-handle
    COMMAND buddy-server-audio-transcription-plugin-handle-test)
  set_tests_properties(
    buddy-server-audio-transcription-plugin-handle PROPERTIES
    PASS_REGULAR_EXPRESSION
      "AudioTranscriptionModelPluginHandle tests passed")

  add_executable(buddy-server-backend-selection-test
    BackendSelectionTest.cpp BackendSelection.cpp)
  target_compile_features(buddy-server-backend-selection-test PRIVATE cxx_std_17)
  target_include_directories(buddy-server-backend-selection-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(buddy-server-backend-selection-test PRIVATE
    buddy_runtime_core)
  add_test(NAME buddy-server-backend-selection
    COMMAND buddy-server-backend-selection-test)
  set_tests_properties(buddy-server-backend-selection PROPERTIES
    PASS_REGULAR_EXPRESSION "BackendSelection tests passed")

  include(${CMAKE_CURRENT_SOURCE_DIR}/TranscriptionTests.cmake)

  add_custom_target(check-buddy-server
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R buddy-server
    DEPENDS
      buddy-server
      buddy-server-json-codec-test
      buddy-server-embedding-plugin-handle-test
      buddy-server-masked-lm-plugin-handle-test
      buddy-server-audio-transcription-plugin-handle-test
      buddy-server-backend-selection-test
      ${BUDDY_SERVER_TRANSCRIPTION_TEST_TARGETS}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()
