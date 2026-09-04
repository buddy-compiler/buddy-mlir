find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Fake file/payload .rax packages used by manifest and HTTP integration tests.

set(_TRANSCRIPTION_FIXTURE_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/transcription-manifest-test")
set(_TRANSCRIPTION_FILE_RAX "${_TRANSCRIPTION_FIXTURE_DIR}/file.rax")
set(_TRANSCRIPTION_PAYLOAD_RAX "${_TRANSCRIPTION_FIXTURE_DIR}/payload.rax")

add_custom_command(
  OUTPUT "${_TRANSCRIPTION_FILE_RAX}" "${_TRANSCRIPTION_PAYLOAD_RAX}"
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
  DEPENDS
    rax-pack
    buddy-server-transcription-fake-plugin
    "${CMAKE_CURRENT_SOURCE_DIR}/TranscriptionManifestTest.mlir"
    "${CMAKE_CURRENT_SOURCE_DIR}/TranscriptionManifestVocab.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/TranscriptionManifestWeights.bin"
  VERBATIM)
add_custom_target(buddy-server-transcription-manifest-fixtures
  DEPENDS "${_TRANSCRIPTION_FILE_RAX}" "${_TRANSCRIPTION_PAYLOAD_RAX}")

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

add_test(NAME buddy-server-manifest-generators
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/ManifestGeneratorTest.py"
          "${CMAKE_SOURCE_DIR}/models/whisper/codegen/gen_whisper_manifest.py"
          "${CMAKE_SOURCE_DIR}/tools/buddy-codegen/gen_manifest.py"
          "${CMAKE_SOURCE_DIR}/models/whisper/specs/base.json"
          "$<TARGET_FILE:rax-pack>"
          "${_TRANSCRIPTION_FIXTURE_DIR}/generator-test")
set_tests_properties(buddy-server-manifest-generators PROPERTIES
  PASS_REGULAR_EXPRESSION "Manifest generator tests passed")

set(BUDDY_SERVER_TRANSCRIPTION_TEST_TARGETS
  buddy-server-transcription-manifest-fixtures
  buddy-server-model-manifest-transcription-test)
