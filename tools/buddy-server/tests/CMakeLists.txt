find_package(Python3 COMPONENTS Interpreter REQUIRED)

function(add_buddy_server_unit_test target test_name source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE buddy-server-core)
  add_test(NAME ${test_name} COMMAND ${target})
endfunction()

function(add_buddy_server_fake_plugin target source)
  add_library(${target} MODULE ${source})
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_link_libraries(${target} PRIVATE buddy_runtime_core)
endfunction()

add_test(NAME buddy-server-help COMMAND buddy-server --help)
set_tests_properties(buddy-server-help PROPERTIES
  PASS_REGULAR_EXPRESSION "buddy-server \\[OPTIONS\\]")

add_buddy_server_unit_test(
  buddy-server-json-codec-test
  buddy-server-json-codec
  unit/JsonCodecTest.cpp)
set_tests_properties(buddy-server-json-codec PROPERTIES
  PASS_REGULAR_EXPRESSION "JsonCodec tests passed")

add_buddy_server_unit_test(
  buddy-server-embedding-plugin-handle-test
  buddy-server-embedding-plugin-handle
  unit/EmbeddingModelPluginHandleTest.cpp)

add_buddy_server_unit_test(
  buddy-server-masked-lm-plugin-handle-test
  buddy-server-masked-lm-plugin-handle
  unit/MaskedLMModelPluginHandleTest.cpp)

add_buddy_server_fake_plugin(
  buddy-server-transcription-fake-plugin
  fixtures/plugins/AudioTranscriptionFakePlugin.cpp)
add_buddy_server_fake_plugin(
  buddy-server-transcription-missing-plugin
  fixtures/plugins/AudioTranscriptionMissingSymbolPlugin.cpp)
add_buddy_server_fake_plugin(
  buddy-server-transcription-null-plugin
  fixtures/plugins/AudioTranscriptionNullPlugin.cpp)

add_buddy_server_unit_test(
  buddy-server-audio-transcription-plugin-handle-test
  buddy-server-audio-transcription-plugin-handle
  unit/AudioTranscriptionModelPluginHandleTest.cpp)
target_compile_definitions(
  buddy-server-audio-transcription-plugin-handle-test PRIVATE
  BUDDY_TEST_TRANSCRIPTION_PLUGIN="$<TARGET_FILE:buddy-server-transcription-fake-plugin>"
  BUDDY_TEST_TRANSCRIPTION_MISSING_PLUGIN="$<TARGET_FILE:buddy-server-transcription-missing-plugin>"
  BUDDY_TEST_TRANSCRIPTION_NULL_PLUGIN="$<TARGET_FILE:buddy-server-transcription-null-plugin>")
add_dependencies(buddy-server-audio-transcription-plugin-handle-test
  buddy-server-transcription-fake-plugin
  buddy-server-transcription-missing-plugin
  buddy-server-transcription-null-plugin)
set_tests_properties(
  buddy-server-audio-transcription-plugin-handle PROPERTIES
  PASS_REGULAR_EXPRESSION
    "AudioTranscriptionModelPluginHandle tests passed")

add_buddy_server_fake_plugin(
  buddy-server-resident-fake-plugin
  fixtures/plugins/ModelServiceFakePlugin.cpp)
add_buddy_server_fake_plugin(
  buddy-server-embedding-fake-plugin
  fixtures/plugins/ModelServiceFakePlugin.cpp)
add_buddy_server_fake_plugin(
  buddy-server-masked-lm-fake-plugin
  fixtures/plugins/ModelServiceFakePlugin.cpp)
set(_buddy_service_fake_plugins
  buddy-server-resident-fake-plugin
  buddy-server-embedding-fake-plugin
  buddy-server-masked-lm-fake-plugin)
target_compile_definitions(buddy-server-resident-fake-plugin PRIVATE
  BUDDY_FAKE_SERVICE_KIND=1)
target_compile_definitions(buddy-server-embedding-fake-plugin PRIVATE
  BUDDY_FAKE_SERVICE_KIND=2)
target_compile_definitions(buddy-server-masked-lm-fake-plugin PRIVATE
  BUDDY_FAKE_SERVICE_KIND=3)

add_buddy_server_unit_test(
  buddy-server-model-plugin-handle-test
  buddy-server-model-plugin-handle
  unit/ServerModelPluginHandleTest.cpp)
target_compile_definitions(
  buddy-server-model-plugin-handle-test PRIVATE
  BUDDY_TEST_SERVER_PLUGIN="$<TARGET_FILE:buddy-server-resident-fake-plugin>"
  BUDDY_TEST_LEGACY_SERVER_PLUGIN="$<TARGET_FILE:buddy-server-discovery-resident-plugin>")

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
foreach(_plugin IN LISTS _buddy_discovery_plugins)
  add_buddy_server_fake_plugin(
    ${_plugin} fixtures/plugins/ModelPluginDiscoveryFakePlugin.cpp)
endforeach()
add_buddy_server_fake_plugin(
  buddy-server-runtime-dependency-fake
  fixtures/plugins/RaxDependencyFakeLibrary.cpp)

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

add_buddy_server_unit_test(
  buddy-server-model-plugin-discovery-test
  buddy-server-model-plugin-discovery
  unit/ModelPluginDiscoveryTest.cpp)
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
add_dependencies(buddy-server-model-plugin-handle-test
  buddy-server-resident-fake-plugin
  buddy-server-discovery-resident-plugin)
set_tests_properties(buddy-server-model-plugin-discovery PROPERTIES
  PASS_REGULAR_EXPRESSION "ModelPluginDiscovery tests passed")

include(${CMAKE_CURRENT_LIST_DIR}/Fixtures.cmake)

add_custom_target(check-buddy-server
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R ^buddy-server
  DEPENDS
    buddy-server
    buddy-server-json-codec-test
    buddy-server-embedding-plugin-handle-test
    buddy-server-masked-lm-plugin-handle-test
    buddy-server-audio-transcription-plugin-handle-test
    buddy-server-model-plugin-discovery-test
    buddy-server-model-plugin-handle-test
    ${BUDDY_SERVER_FIXTURE_TARGETS}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
