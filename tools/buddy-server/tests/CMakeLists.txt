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

add_buddy_server_fake_plugin(
  buddy-server-transcription-fake-plugin
  fixtures/plugins/AudioTranscriptionFakePlugin.cpp)

add_buddy_server_fake_plugin(
  buddy-server-resident-fake-plugin
  fixtures/plugins/ModelServiceFakePlugin.cpp)
add_buddy_server_fake_plugin(
  buddy-server-embedding-fake-plugin
  fixtures/plugins/ModelServiceFakePlugin.cpp)
add_buddy_server_fake_plugin(
  buddy-server-masked-lm-fake-plugin
  fixtures/plugins/ModelServiceFakePlugin.cpp)
add_buddy_server_fake_plugin(
  buddy-server-non-streaming-fake-plugin
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
target_compile_definitions(buddy-server-non-streaming-fake-plugin PRIVATE
  BUDDY_FAKE_SERVICE_KIND=1
  BUDDY_FAKE_DISABLE_STREAMING=1)

add_buddy_server_fake_plugin(
  buddy-server-legacy-model-plugin
  fixtures/plugins/LegacyServerModelPlugin.cpp)
add_buddy_server_fake_plugin(
  buddy-server-dependency-model-plugin
  fixtures/plugins/ServerModelDependencyPlugin.cpp)
target_link_libraries(buddy-server-dependency-model-plugin PRIVATE
  ${CMAKE_DL_LIBS})
add_buddy_server_fake_plugin(
  buddy-server-runtime-dependency-fake
  fixtures/plugins/RaxDependencyFakeLibrary.cpp)
add_buddy_server_fake_plugin(
  buddy-server-abi-error-fake-plugin
  fixtures/plugins/ServerModelErrorFakePlugin.cpp)
add_buddy_server_fake_plugin(
  buddy-server-non-thread-safe-fake-plugin
  fixtures/plugins/ServerModelConcurrencyFakePlugin.cpp)
add_buddy_server_fake_plugin(
  buddy-server-thread-safe-fake-plugin
  fixtures/plugins/ServerModelConcurrencyFakePlugin.cpp)
target_compile_definitions(buddy-server-thread-safe-fake-plugin PRIVATE
  BUDDY_FAKE_THREAD_SAFE=1)

add_buddy_server_unit_test(
  buddy-server-model-plugin-handle-test
  buddy-server-model-plugin-handle
  unit/ServerModelPluginHandleTest.cpp)
target_compile_definitions(
  buddy-server-model-plugin-handle-test PRIVATE
  BUDDY_TEST_SERVER_PLUGIN="$<TARGET_FILE:buddy-server-resident-fake-plugin>"
  BUDDY_TEST_NON_STREAMING_SERVER_PLUGIN="$<TARGET_FILE:buddy-server-non-streaming-fake-plugin>"
  BUDDY_TEST_NON_THREAD_SAFE_SERVER_PLUGIN="$<TARGET_FILE:buddy-server-non-thread-safe-fake-plugin>"
  BUDDY_TEST_THREAD_SAFE_SERVER_PLUGIN="$<TARGET_FILE:buddy-server-thread-safe-fake-plugin>"
  BUDDY_TEST_LEGACY_SERVER_PLUGIN="$<TARGET_FILE:buddy-server-legacy-model-plugin>")
add_dependencies(buddy-server-model-plugin-handle-test
  buddy-server-resident-fake-plugin
  buddy-server-non-streaming-fake-plugin
  buddy-server-non-thread-safe-fake-plugin
  buddy-server-thread-safe-fake-plugin
  buddy-server-legacy-model-plugin)

include(${CMAKE_CURRENT_LIST_DIR}/Fixtures.cmake)

add_custom_target(check-buddy-server
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R ^buddy-server
  DEPENDS
    buddy-server
    buddy-server-json-codec-test
    buddy-server-model-plugin-handle-test
    ${BUDDY_SERVER_FIXTURE_TARGETS}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
