# tools/buddy-server/CMakeLists.txt

find_package(Threads REQUIRED)

set(BUDDY_SERVER_COMMON_SOURCES
  JsonCodec.cpp
  AudioTranscriptionJsonCodec.cpp
  ResidentModelPluginHandle.cpp
  EmbeddingModelPluginHandle.cpp
  MaskedLMModelPluginHandle.cpp
  AudioTranscriptionModelPluginHandle.cpp
  ModelPluginDiscovery.cpp
  SimpleHttpServer.cpp
)

add_executable(buddy-server
  buddy-server.cpp
  ${BUDDY_SERVER_COMMON_SOURCES}
)
target_compile_features(buddy-server PRIVATE cxx_std_17)
target_compile_definitions(buddy-server PRIVATE
  BUDDY_VERSION="${BUDDY_PACKAGE_VERSION}")
target_include_directories(buddy-server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(buddy-server PRIVATE
  CLI11
  buddy_runtime_core
  buddy_runtime_llm
  buddy_runtime_rax_dependencies
  LLVMSupport
  Threads::Threads
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
    PASS_REGULAR_EXPRESSION "buddy-server \\[OPTIONS\\]")

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

  add_library(buddy-server-resident-fake-plugin MODULE
    ModelServiceFakePlugin.cpp)
  add_library(buddy-server-embedding-fake-plugin MODULE
    ModelServiceFakePlugin.cpp)
  add_library(buddy-server-masked-lm-fake-plugin MODULE
    ModelServiceFakePlugin.cpp)
  set(_buddy_service_fake_plugins
    buddy-server-resident-fake-plugin
    buddy-server-embedding-fake-plugin
    buddy-server-masked-lm-fake-plugin)
  foreach(_plugin ${_buddy_service_fake_plugins})
    target_compile_features(${_plugin} PRIVATE cxx_std_17)
    target_link_libraries(${_plugin} PRIVATE buddy_runtime_core)
  endforeach()
  target_compile_definitions(buddy-server-resident-fake-plugin PRIVATE
    BUDDY_FAKE_SERVICE_KIND=1)
  target_compile_definitions(buddy-server-embedding-fake-plugin PRIVATE
    BUDDY_FAKE_SERVICE_KIND=2)
  target_compile_definitions(buddy-server-masked-lm-fake-plugin PRIVATE
    BUDDY_FAKE_SERVICE_KIND=3)

  add_library(buddy-server-discovery-resident-plugin MODULE
    ModelPluginDiscoveryFakePlugin.cpp)
  add_library(buddy-server-discovery-embedding-plugin MODULE
    ModelPluginDiscoveryFakePlugin.cpp)
  add_library(buddy-server-discovery-masked-lm-plugin MODULE
    ModelPluginDiscoveryFakePlugin.cpp)
  add_library(buddy-server-discovery-transcription-plugin MODULE
    ModelPluginDiscoveryFakePlugin.cpp)
  add_library(buddy-server-discovery-missing-plugin MODULE
    ModelPluginDiscoveryFakePlugin.cpp)
  add_library(buddy-server-discovery-null-plugin MODULE
    ModelPluginDiscoveryFakePlugin.cpp)
  add_library(buddy-server-discovery-version-plugin MODULE
    ModelPluginDiscoveryFakePlugin.cpp)
  add_library(buddy-server-discovery-small-plugin MODULE
    ModelPluginDiscoveryFakePlugin.cpp)
  add_library(buddy-server-discovery-reserved-plugin MODULE
    ModelPluginDiscoveryFakePlugin.cpp)
  add_library(buddy-server-discovery-unknown-plugin MODULE
    ModelPluginDiscoveryFakePlugin.cpp)
  add_library(buddy-server-discovery-dependent-plugin MODULE
    ModelPluginDiscoveryFakePlugin.cpp)
  add_library(buddy-server-runtime-dependency-fake MODULE
    RaxDependencyFakeLibrary.cpp)
  set(_buddy_discovery_plugins
    buddy-server-discovery-resident-plugin
    buddy-server-discovery-embedding-plugin
    buddy-server-discovery-masked-lm-plugin
    buddy-server-discovery-transcription-plugin
    buddy-server-discovery-missing-plugin
    buddy-server-discovery-null-plugin
    buddy-server-discovery-version-plugin
    buddy-server-discovery-small-plugin
    buddy-server-discovery-reserved-plugin
    buddy-server-discovery-unknown-plugin
    buddy-server-discovery-dependent-plugin)
  foreach(_plugin ${_buddy_discovery_plugins})
    target_compile_features(${_plugin} PRIVATE cxx_std_17)
    target_link_libraries(${_plugin} PRIVATE buddy_runtime_core)
  endforeach()
  target_compile_features(buddy-server-runtime-dependency-fake PRIVATE
    cxx_std_17)
  target_compile_definitions(buddy-server-discovery-resident-plugin PRIVATE
    BUDDY_DISCOVERY_SERVICE_KIND=1)
  target_compile_definitions(buddy-server-discovery-embedding-plugin PRIVATE
    BUDDY_DISCOVERY_SERVICE_KIND=2)
  target_compile_definitions(buddy-server-discovery-masked-lm-plugin PRIVATE
    BUDDY_DISCOVERY_SERVICE_KIND=3)
  target_compile_definitions(buddy-server-discovery-transcription-plugin PRIVATE
    BUDDY_DISCOVERY_SERVICE_KIND=4)
  target_compile_definitions(buddy-server-discovery-missing-plugin PRIVATE
    BUDDY_DISCOVERY_MISSING_DESCRIPTOR=1)
  target_compile_definitions(buddy-server-discovery-null-plugin PRIVATE
    BUDDY_DISCOVERY_NULL_DESCRIPTOR=1)
  target_compile_definitions(buddy-server-discovery-version-plugin PRIVATE
    BUDDY_DISCOVERY_ABI_VERSION=2)
  target_compile_definitions(buddy-server-discovery-small-plugin PRIVATE
    BUDDY_DISCOVERY_STRUCT_SIZE=4)
  target_compile_definitions(buddy-server-discovery-reserved-plugin PRIVATE
    BUDDY_DISCOVERY_SERVICE_KIND=1 BUDDY_DISCOVERY_RESERVED=1)
  target_compile_definitions(buddy-server-discovery-unknown-plugin PRIVATE
    BUDDY_DISCOVERY_SERVICE_KIND=99)
  target_compile_definitions(buddy-server-discovery-dependent-plugin PRIVATE
    BUDDY_DISCOVERY_SERVICE_KIND=1
    BUDDY_DISCOVERY_REQUIRED_GLOBAL_SYMBOL=1)
  target_link_libraries(buddy-server-discovery-dependent-plugin PRIVATE
    ${CMAKE_DL_LIBS})

  add_executable(buddy-server-model-plugin-discovery-test
    ModelPluginDiscoveryTest.cpp ModelPluginDiscovery.cpp)
  target_compile_features(buddy-server-model-plugin-discovery-test PRIVATE
    cxx_std_17)
  target_include_directories(buddy-server-model-plugin-discovery-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(buddy-server-model-plugin-discovery-test PRIVATE
    buddy_runtime_core ${CMAKE_DL_LIBS})
  target_compile_definitions(buddy-server-model-plugin-discovery-test PRIVATE
    BUDDY_DISCOVERY_RESIDENT_PLUGIN="$<TARGET_FILE:buddy-server-discovery-resident-plugin>"
    BUDDY_DISCOVERY_EMBEDDING_PLUGIN="$<TARGET_FILE:buddy-server-discovery-embedding-plugin>"
    BUDDY_DISCOVERY_MASKED_LM_PLUGIN="$<TARGET_FILE:buddy-server-discovery-masked-lm-plugin>"
    BUDDY_DISCOVERY_TRANSCRIPTION_PLUGIN="$<TARGET_FILE:buddy-server-discovery-transcription-plugin>"
    BUDDY_DISCOVERY_MISSING_PLUGIN="$<TARGET_FILE:buddy-server-discovery-missing-plugin>"
    BUDDY_DISCOVERY_NULL_PLUGIN="$<TARGET_FILE:buddy-server-discovery-null-plugin>"
    BUDDY_DISCOVERY_VERSION_PLUGIN="$<TARGET_FILE:buddy-server-discovery-version-plugin>"
    BUDDY_DISCOVERY_SMALL_PLUGIN="$<TARGET_FILE:buddy-server-discovery-small-plugin>"
    BUDDY_DISCOVERY_RESERVED_PLUGIN="$<TARGET_FILE:buddy-server-discovery-reserved-plugin>"
    BUDDY_DISCOVERY_UNKNOWN_PLUGIN="$<TARGET_FILE:buddy-server-discovery-unknown-plugin>")
  add_dependencies(buddy-server-model-plugin-discovery-test
    ${_buddy_discovery_plugins})
  add_test(NAME buddy-server-model-plugin-discovery
    COMMAND buddy-server-model-plugin-discovery-test)
  set_tests_properties(buddy-server-model-plugin-discovery PROPERTIES
    PASS_REGULAR_EXPRESSION "ModelPluginDiscovery tests passed")

  include(${CMAKE_CURRENT_SOURCE_DIR}/TranscriptionTests.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/ResidentAdapterChecks.cmake)

  add_custom_target(check-buddy-server
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j1 -R buddy-server
    DEPENDS
      buddy-server
      buddy-server-json-codec-test
      buddy-server-embedding-plugin-handle-test
      buddy-server-masked-lm-plugin-handle-test
      buddy-server-audio-transcription-plugin-handle-test
      buddy-server-model-plugin-discovery-test
      ${BUDDY_SERVER_TRANSCRIPTION_TEST_TARGETS}
      ${BUDDY_SERVER_RESIDENT_ADAPTER_TARGETS}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()
