find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Fake file/payload .rax packages used by manifest and HTTP integration tests.

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
          "${CMAKE_CURRENT_SOURCE_DIR}/TranscriptionManifestVocab.txt"
          "${_TRANSCRIPTION_FIXTURE_DIR}/vocab.txt"
  COMMAND ${CMAKE_COMMAND} -E copy
          "${CMAKE_CURRENT_SOURCE_DIR}/TranscriptionManifestWeights.bin"
          "${_TRANSCRIPTION_FIXTURE_DIR}/weights.bin"
  COMMAND $<TARGET_FILE:rax-pack>
          "${CMAKE_CURRENT_SOURCE_DIR}/TranscriptionManifestTest.mlir"
          -o "${_TRANSCRIPTION_FILE_RAX}"
  COMMAND $<TARGET_FILE:rax-pack>
          "${CMAKE_CURRENT_SOURCE_DIR}/TranscriptionManifestTest.mlir"
          -o "${_TRANSCRIPTION_PAYLOAD_RAX}" --embed-payload
  COMMAND $<TARGET_FILE:rax-pack>
          "${CMAKE_CURRENT_SOURCE_DIR}/NoModelLibraryManifestTest.mlir"
          -o "${_NO_MODEL_LIBRARY_RAX}"
  DEPENDS
    rax-pack
    buddy-server-transcription-fake-plugin
    "${CMAKE_CURRENT_SOURCE_DIR}/TranscriptionManifestTest.mlir"
    "${CMAKE_CURRENT_SOURCE_DIR}/NoModelLibraryManifestTest.mlir"
    "${CMAKE_CURRENT_SOURCE_DIR}/TranscriptionManifestVocab.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/TranscriptionManifestWeights.bin"
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
          "${CMAKE_CURRENT_SOURCE_DIR}/ModelServiceManifestTest.mlir"
          -o "${_RESIDENT_SERVICE_RAX}"
  COMMAND $<TARGET_FILE:rax-pack>
          "${CMAKE_CURRENT_SOURCE_DIR}/ModelServiceManifestTest.mlir"
          -o "${_EMBEDDING_SERVICE_RAX}"
  COMMAND $<TARGET_FILE:rax-pack>
          "${CMAKE_CURRENT_SOURCE_DIR}/ModelServiceManifestTest.mlir"
          -o "${_MASKED_LM_SERVICE_RAX}"
  DEPENDS
    rax-pack
    ${_buddy_service_fake_plugins}
    "${CMAKE_CURRENT_SOURCE_DIR}/ModelServiceManifestTest.mlir"
  VERBATIM)
add_custom_target(buddy-server-model-service-fixtures
  DEPENDS
    "${_RESIDENT_SERVICE_RAX}"
    "${_EMBEDDING_SERVICE_RAX}"
    "${_MASKED_LM_SERVICE_RAX}")

add_executable(buddy-server-model-manifest-transcription-test
  ModelManifestTranscriptionTest.cpp)
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
  ModelManifestDuplicateAttributeTest.cpp)
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
          "${CMAKE_CURRENT_SOURCE_DIR}/BuddyServerTranscriptionIntegrationTest.py"
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
          "${CMAKE_CURRENT_SOURCE_DIR}/BuddyServerServiceIntegrationTest.py"
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
          "${CMAKE_CURRENT_SOURCE_DIR}/BuddyServerStartupFailureTest.py"
          "$<TARGET_FILE:buddy-server>"
          "${_TRANSCRIPTION_FILE_RAX}"
          "${_NO_MODEL_LIBRARY_RAX}"
          "$<TARGET_FILE:buddy-server-discovery-resident-plugin>"
          "$<TARGET_FILE:buddy-server-discovery-missing-plugin>"
          "${CMAKE_CURRENT_BINARY_DIR}/startup-failure-test")
set_tests_properties(buddy-server-startup-failures PROPERTIES
  TIMEOUT 20
  PASS_REGULAR_EXPRESSION "buddy-server startup failure tests passed")

add_test(NAME buddy-server-manifest-generators
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/ManifestGeneratorTest.py"
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

set(BUDDY_SERVER_TRANSCRIPTION_TEST_TARGETS
  buddy-server-transcription-manifest-fixtures
  buddy-server-model-service-fixtures
  buddy-server-model-manifest-transcription-test
  buddy-server-model-manifest-duplicate-attribute-test)
