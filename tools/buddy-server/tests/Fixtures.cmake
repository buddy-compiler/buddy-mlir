find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Fake file/payload .rax packages used by manifest and HTTP integration tests.

set(_BUDDY_SERVER_TEST_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(_BUDDY_SERVER_MANIFEST_DIR
  "${_BUDDY_SERVER_TEST_SOURCE_DIR}/fixtures/manifests")
set(_BUDDY_SERVER_DATA_DIR
  "${_BUDDY_SERVER_TEST_SOURCE_DIR}/fixtures/data")
set(_BUDDY_SERVER_INTEGRATION_DIR
  "${_BUDDY_SERVER_TEST_SOURCE_DIR}/integration")
set(_BUDDY_SERVER_PACKAGING_DIR
  "${_BUDDY_SERVER_TEST_SOURCE_DIR}/packaging")

set(_TRANSCRIPTION_FIXTURE_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/transcription-manifest-test")
set(_TRANSCRIPTION_FILE_RAX "${_TRANSCRIPTION_FIXTURE_DIR}/file.rax")
set(_TRANSCRIPTION_PAYLOAD_RAX "${_TRANSCRIPTION_FIXTURE_DIR}/payload.rax")
set(_NO_MODEL_LIBRARY_RAX "${_TRANSCRIPTION_FIXTURE_DIR}/no-model-library.rax")
set(_MODEL_SERVICE_FIXTURE_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/model-service-test")
set(_RESIDENT_SERVICE_RAX "${_MODEL_SERVICE_FIXTURE_DIR}/resident/service.rax")
set(_EMBEDDING_SERVICE_RAX
  "${_MODEL_SERVICE_FIXTURE_DIR}/embedding/service.rax")
set(_MASKED_LM_SERVICE_RAX
  "${_MODEL_SERVICE_FIXTURE_DIR}/masked-lm/service.rax")
set(_RUNTIME_DEPENDENCY_FIXTURE_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/runtime-dependency-test")
set(_RUNTIME_DEPENDENCY_RAX "${_RUNTIME_DEPENDENCY_FIXTURE_DIR}/service.rax")
set(_ABI_ERROR_FIXTURE_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/abi-error-test")
set(_ABI_ERROR_RAX "${_ABI_ERROR_FIXTURE_DIR}/service.rax")
set(_NON_STREAMING_FIXTURE_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/non-streaming-test")
set(_NON_STREAMING_RAX "${_NON_STREAMING_FIXTURE_DIR}/service.rax")

add_custom_command(
  OUTPUT
    "${_TRANSCRIPTION_FILE_RAX}"
    "${_TRANSCRIPTION_PAYLOAD_RAX}"
    "${_NO_MODEL_LIBRARY_RAX}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_TRANSCRIPTION_FIXTURE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy
          "$<TARGET_FILE:buddy-server-transcription-fake-plugin>"
          "${_TRANSCRIPTION_FIXTURE_DIR}/model.so"
  COMMAND ${CMAKE_COMMAND} -E copy
          "$<TARGET_FILE:buddy-server-transcription-fake-plugin>"
          "${_TRANSCRIPTION_FIXTURE_DIR}/runner.so"
  COMMAND ${CMAKE_COMMAND} -E copy
          "$<TARGET_FILE:buddy-server-transcription-fake-plugin>"
          "${_TRANSCRIPTION_FIXTURE_DIR}/transcription.so"
  COMMAND ${CMAKE_COMMAND} -E copy
          "${_BUDDY_SERVER_DATA_DIR}/TranscriptionManifestVocab.txt"
          "${_TRANSCRIPTION_FIXTURE_DIR}/vocab.txt"
  COMMAND ${CMAKE_COMMAND} -E copy
          "${_BUDDY_SERVER_DATA_DIR}/TranscriptionManifestWeights.bin"
          "${_TRANSCRIPTION_FIXTURE_DIR}/weights.bin"
  COMMAND $<TARGET_FILE:rax-pack>
          "${_BUDDY_SERVER_MANIFEST_DIR}/TranscriptionManifestTest.mlir"
          -o "${_TRANSCRIPTION_FILE_RAX}"
  COMMAND $<TARGET_FILE:rax-pack>
          "${_BUDDY_SERVER_MANIFEST_DIR}/TranscriptionManifestTest.mlir"
          -o "${_TRANSCRIPTION_PAYLOAD_RAX}" --embed-payload
  COMMAND $<TARGET_FILE:rax-pack>
          "${_BUDDY_SERVER_MANIFEST_DIR}/NoModelLibraryManifestTest.mlir"
          -o "${_NO_MODEL_LIBRARY_RAX}"
  DEPENDS
    rax-pack
    buddy-server-transcription-fake-plugin
    "${_BUDDY_SERVER_MANIFEST_DIR}/TranscriptionManifestTest.mlir"
    "${_BUDDY_SERVER_MANIFEST_DIR}/NoModelLibraryManifestTest.mlir"
    "${_BUDDY_SERVER_DATA_DIR}/TranscriptionManifestVocab.txt"
    "${_BUDDY_SERVER_DATA_DIR}/TranscriptionManifestWeights.bin"
  VERBATIM)
add_custom_target(buddy-server-transcription-manifest-fixtures
  DEPENDS
    "${_TRANSCRIPTION_FILE_RAX}"
    "${_TRANSCRIPTION_PAYLOAD_RAX}"
    "${_NO_MODEL_LIBRARY_RAX}")

add_custom_command(
  OUTPUT
    "${_RESIDENT_SERVICE_RAX}"
    "${_EMBEDDING_SERVICE_RAX}"
    "${_MASKED_LM_SERVICE_RAX}"
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${_MODEL_SERVICE_FIXTURE_DIR}/resident"
          "${_MODEL_SERVICE_FIXTURE_DIR}/embedding"
          "${_MODEL_SERVICE_FIXTURE_DIR}/masked-lm"
  COMMAND ${CMAKE_COMMAND} -E copy
          "$<TARGET_FILE:buddy-server-resident-fake-plugin>"
          "${_MODEL_SERVICE_FIXTURE_DIR}/resident/service.so"
  COMMAND ${CMAKE_COMMAND} -E copy
          "$<TARGET_FILE:buddy-server-embedding-fake-plugin>"
          "${_MODEL_SERVICE_FIXTURE_DIR}/embedding/service.so"
  COMMAND ${CMAKE_COMMAND} -E copy
          "$<TARGET_FILE:buddy-server-masked-lm-fake-plugin>"
          "${_MODEL_SERVICE_FIXTURE_DIR}/masked-lm/service.so"
  COMMAND $<TARGET_FILE:rax-pack>
          "${_BUDDY_SERVER_MANIFEST_DIR}/ModelServiceManifestTest.mlir"
          -o "${_RESIDENT_SERVICE_RAX}"
  COMMAND $<TARGET_FILE:rax-pack>
          "${_BUDDY_SERVER_MANIFEST_DIR}/ModelServiceManifestTest.mlir"
          -o "${_EMBEDDING_SERVICE_RAX}"
  COMMAND $<TARGET_FILE:rax-pack>
          "${_BUDDY_SERVER_MANIFEST_DIR}/ModelServiceManifestTest.mlir"
          -o "${_MASKED_LM_SERVICE_RAX}"
  DEPENDS
    rax-pack
    ${_buddy_service_fake_plugins}
    "${_BUDDY_SERVER_MANIFEST_DIR}/ModelServiceManifestTest.mlir"
  VERBATIM)
add_custom_target(buddy-server-model-service-fixtures
  DEPENDS
    "${_RESIDENT_SERVICE_RAX}"
    "${_EMBEDDING_SERVICE_RAX}"
    "${_MASKED_LM_SERVICE_RAX}")

add_custom_command(
  OUTPUT "${_RUNTIME_DEPENDENCY_RAX}"
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${_RUNTIME_DEPENDENCY_FIXTURE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy
          "$<TARGET_FILE:buddy-server-runtime-dependency-fake>"
          "${_RUNTIME_DEPENDENCY_FIXTURE_DIR}/model.so"
  COMMAND ${CMAKE_COMMAND} -E copy
          "$<TARGET_FILE:buddy-server-runtime-dependency-fake>"
          "${_RUNTIME_DEPENDENCY_FIXTURE_DIR}/dependency.so"
  COMMAND ${CMAKE_COMMAND} -E copy
          "$<TARGET_FILE:buddy-server-dependency-model-plugin>"
          "${_RUNTIME_DEPENDENCY_FIXTURE_DIR}/service.so"
  COMMAND $<TARGET_FILE:rax-pack>
          "${_BUDDY_SERVER_MANIFEST_DIR}/RuntimeDependencyManifestTest.mlir"
          -o "${_RUNTIME_DEPENDENCY_RAX}"
  DEPENDS
    rax-pack
    buddy-server-runtime-dependency-fake
    buddy-server-dependency-model-plugin
    "${_BUDDY_SERVER_MANIFEST_DIR}/RuntimeDependencyManifestTest.mlir"
  VERBATIM)
add_custom_target(buddy-server-runtime-dependency-fixture
  DEPENDS "${_RUNTIME_DEPENDENCY_RAX}")

add_custom_command(
  OUTPUT "${_ABI_ERROR_RAX}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_ABI_ERROR_FIXTURE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy
          "$<TARGET_FILE:buddy-server-abi-error-fake-plugin>"
          "${_ABI_ERROR_FIXTURE_DIR}/service.so"
  COMMAND $<TARGET_FILE:rax-pack>
          "${_BUDDY_SERVER_MANIFEST_DIR}/ModelServiceManifestTest.mlir"
          -o "${_ABI_ERROR_RAX}"
  DEPENDS
    rax-pack
    buddy-server-abi-error-fake-plugin
    "${_BUDDY_SERVER_MANIFEST_DIR}/ModelServiceManifestTest.mlir"
  VERBATIM)
add_custom_target(buddy-server-abi-error-fixture
  DEPENDS "${_ABI_ERROR_RAX}")

add_custom_command(
  OUTPUT "${_NON_STREAMING_RAX}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_NON_STREAMING_FIXTURE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy
          "$<TARGET_FILE:buddy-server-non-streaming-fake-plugin>"
          "${_NON_STREAMING_FIXTURE_DIR}/service.so"
  COMMAND $<TARGET_FILE:rax-pack>
          "${_BUDDY_SERVER_MANIFEST_DIR}/ModelServiceManifestTest.mlir"
          -o "${_NON_STREAMING_RAX}"
  DEPENDS
    rax-pack
    buddy-server-non-streaming-fake-plugin
    "${_BUDDY_SERVER_MANIFEST_DIR}/ModelServiceManifestTest.mlir"
  VERBATIM)
add_custom_target(buddy-server-non-streaming-fixture
  DEPENDS "${_NON_STREAMING_RAX}")

add_executable(buddy-server-model-manifest-transcription-test
  packaging/ModelManifestTranscriptionTest.cpp)
target_compile_features(buddy-server-model-manifest-transcription-test
  PRIVATE cxx_std_17)
target_link_libraries(buddy-server-model-manifest-transcription-test PRIVATE
  buddy_runtime_core)
add_dependencies(buddy-server-model-manifest-transcription-test
  buddy-server-transcription-manifest-fixtures)
add_test(NAME buddy-server-model-manifest-transcription
  COMMAND buddy-server-model-manifest-transcription-test
          "${_TRANSCRIPTION_FILE_RAX}" "${_TRANSCRIPTION_PAYLOAD_RAX}")
set_tests_properties(buddy-server-model-manifest-transcription PROPERTIES
  ENVIRONMENT
    "BUDDY_RAX_PAYLOAD_DIR=${_TRANSCRIPTION_FIXTURE_DIR}/payload-cache"
  PASS_REGULAR_EXPRESSION "ModelManifest transcription tests passed")

add_executable(buddy-server-model-manifest-duplicate-attribute-test
  packaging/ModelManifestDuplicateAttributeTest.cpp)
target_compile_features(
  buddy-server-model-manifest-duplicate-attribute-test PRIVATE cxx_std_17)
target_link_libraries(
  buddy-server-model-manifest-duplicate-attribute-test PRIVATE
  buddy_runtime_core)
add_test(NAME buddy-server-model-manifest-duplicate-attribute
  COMMAND buddy-server-model-manifest-duplicate-attribute-test
          "${_TRANSCRIPTION_FIXTURE_DIR}/duplicate-test")
set_tests_properties(
  buddy-server-model-manifest-duplicate-attribute PROPERTIES
  PASS_REGULAR_EXPRESSION "ModelManifest duplicate attribute test passed")

add_test(NAME buddy-server-transcription-http-integration
  COMMAND "${Python3_EXECUTABLE}"
          "${_BUDDY_SERVER_INTEGRATION_DIR}/BuddyServerTranscriptionIntegrationTest.py"
          "$<TARGET_FILE:buddy-server>"
          "${_TRANSCRIPTION_PAYLOAD_RAX}"
          18991
          "${_TRANSCRIPTION_FIXTURE_DIR}/http-payload-cache")
set_tests_properties(buddy-server-transcription-http-integration PROPERTIES
  TIMEOUT 20
  PASS_REGULAR_EXPRESSION
    "buddy-server transcription integration tests passed")

add_test(NAME buddy-server-model-service-http-integration
  COMMAND "${Python3_EXECUTABLE}"
          "${_BUDDY_SERVER_INTEGRATION_DIR}/BuddyServerServiceIntegrationTest.py"
          "$<TARGET_FILE:buddy-server>"
          "${_RESIDENT_SERVICE_RAX}"
          "${_EMBEDDING_SERVICE_RAX}"
          "${_MASKED_LM_SERVICE_RAX}")
set_tests_properties(buddy-server-model-service-http-integration PROPERTIES
  TIMEOUT 30
  PASS_REGULAR_EXPRESSION
    "buddy-server model service integration tests passed")

add_test(NAME buddy-server-startup-failures
  COMMAND "${Python3_EXECUTABLE}"
          "${_BUDDY_SERVER_INTEGRATION_DIR}/BuddyServerStartupFailureTest.py"
          "$<TARGET_FILE:buddy-server>"
          "${_TRANSCRIPTION_FILE_RAX}"
          "${_NO_MODEL_LIBRARY_RAX}"
          "$<TARGET_FILE:buddy-server-legacy-model-plugin>"
          "$<TARGET_FILE:buddy-server-runtime-dependency-fake>"
          "${CMAKE_CURRENT_BINARY_DIR}/startup-failure-test")
set_tests_properties(buddy-server-startup-failures PROPERTIES
  TIMEOUT 20
  PASS_REGULAR_EXPRESSION "buddy-server startup failure tests passed")

add_test(NAME buddy-server-runtime-dependencies
  COMMAND "${Python3_EXECUTABLE}"
          "${_BUDDY_SERVER_INTEGRATION_DIR}/BuddyServerRaxDependencyTest.py"
          "$<TARGET_FILE:buddy-server>"
          "${_RUNTIME_DEPENDENCY_RAX}")
set_tests_properties(buddy-server-runtime-dependencies PROPERTIES
  TIMEOUT 20
  PASS_REGULAR_EXPRESSION
    "buddy-server RAX dependency test passed")

add_test(NAME buddy-server-abi-http-errors
  COMMAND "${Python3_EXECUTABLE}"
          "${_BUDDY_SERVER_INTEGRATION_DIR}/BuddyServerAbiErrorIntegrationTest.py"
          "$<TARGET_FILE:buddy-server>"
          "${_ABI_ERROR_RAX}"
          18997)
set_tests_properties(buddy-server-abi-http-errors PROPERTIES
  TIMEOUT 20
  PASS_REGULAR_EXPRESSION
    "buddy-server ABI HTTP error integration tests passed")

add_test(NAME buddy-server-streaming-capability
  COMMAND "${Python3_EXECUTABLE}"
          "${_BUDDY_SERVER_INTEGRATION_DIR}/BuddyServerCapabilityIntegrationTest.py"
          "$<TARGET_FILE:buddy-server>"
          "${_NON_STREAMING_RAX}"
          18999)
set_tests_properties(buddy-server-streaming-capability PROPERTIES
  TIMEOUT 20
  PASS_REGULAR_EXPRESSION
    "buddy-server streaming capability integration tests passed")

add_test(NAME buddy-server-manifest-generators
  COMMAND "${Python3_EXECUTABLE}"
          "${_BUDDY_SERVER_PACKAGING_DIR}/ManifestGeneratorTest.py"
          "${CMAKE_SOURCE_DIR}/models/whisper/codegen/gen_whisper_manifest.py"
          "${CMAKE_SOURCE_DIR}/tools/buddy-codegen/gen_manifest.py"
          "${CMAKE_SOURCE_DIR}/models/bge_m3/codegen/gen_bge_m3_manifest.py"
          "${CMAKE_SOURCE_DIR}/models/proteinglm/codegen/gen_proteinglm_manifest.py"
          "${CMAKE_SOURCE_DIR}/tools/buddy-codegen/gen_tenstorrent_manifest.py"
          "${CMAKE_SOURCE_DIR}/models/whisper/specs/base.json"
          "$<TARGET_FILE:rax-pack>"
          "${_TRANSCRIPTION_FIXTURE_DIR}/generator-test")
set_tests_properties(buddy-server-manifest-generators PROPERTIES
  PASS_REGULAR_EXPRESSION "Manifest generator tests passed")

set(BUDDY_SERVER_FIXTURE_TARGETS
  buddy-server-transcription-manifest-fixtures
  buddy-server-model-service-fixtures
  buddy-server-runtime-dependency-fixture
  buddy-server-abi-error-fixture
  buddy-server-non-streaming-fixture
  buddy-server-model-manifest-transcription-test
  buddy-server-model-manifest-duplicate-attribute-test)
