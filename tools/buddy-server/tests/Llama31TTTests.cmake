# Exercise production TT execution/resident code without an SDK or device.
# Programs, weights, vocabulary and selected tokens are synthetic fixtures.
set(_TT_MODEL_DIR "${BUDDY_SOURCE_DIR}/models/llama31_tt")
set(_TT_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/llama31-tt-test")
set(_TT_TEST_RAX_FILES)
file(GLOB_RECURSE _TT_ARTIFACT_FILES CONFIGURE_DEPENDS
  "${_TT_MODEL_DIR}/testdata/artifacts/*")
foreach(_profile legacy batch serving)
  set(_batch 1)
  set(_context 64)
  set(_tokenizer "${_TT_MODEL_DIR}/testdata/tokenizer")
  if(_profile STREQUAL "batch")
    set(_batch 4)
  elseif(_profile STREQUAL "serving")
    set(_context 128)
    set(_tokenizer "${_TT_MODEL_DIR}/testdata/serving-tokenizer")
  endif()
  set(_rax "${_TT_TEST_DIR}/${_profile}.rax")
  file(GLOB _tokenizer_files CONFIGURE_DEPENDS "${_tokenizer}/*")
  add_custom_command(OUTPUT "${_rax}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_TT_TEST_DIR}"
    COMMAND "${Python3_EXECUTABLE}"
      "${BUDDY_SOURCE_DIR}/tools/buddy-codegen/gen_tenstorrent_manifest.py"
      --model-name llama31_tt --prefill-ttnn "${_TT_MODEL_DIR}/testdata/prefill.ttnn"
      --decode-ttnn "${_TT_MODEL_DIR}/testdata/decode.ttnn"
      --artifacts "${_TT_MODEL_DIR}/testdata/artifacts"
      --tokenizer "${_tokenizer}" --max-cache-len ${_context}
      --batch-size ${_batch} -o "${_TT_TEST_DIR}/${_profile}.mlir"
    COMMAND $<TARGET_FILE:rax-pack> "${_TT_TEST_DIR}/${_profile}.mlir" -o "${_rax}"
    DEPENDS rax-pack
      "${BUDDY_SOURCE_DIR}/tools/buddy-codegen/gen_tenstorrent_manifest.py"
      "${_TT_MODEL_DIR}/testdata/prefill.ttnn"
      "${_TT_MODEL_DIR}/testdata/decode.ttnn"
      ${_TT_ARTIFACT_FILES}
      ${_tokenizer_files}
    VERBATIM)
  list(APPEND _TT_TEST_RAX_FILES "${_rax}")
endforeach()
add_custom_target(buddy-server-llama31-tt-test-fixtures
  DEPENDS ${_TT_TEST_RAX_FILES})

foreach(_test execution batch-manifest serving)
  if(_test STREQUAL "execution")
    set(_source "${_TT_MODEL_DIR}/Llama31TTExecutionTest.cpp")
  elseif(_test STREQUAL "batch-manifest")
    set(_source "${_TT_MODEL_DIR}/Llama31TTBatchManifestTest.cpp")
  else()
    set(_source "${CMAKE_CURRENT_LIST_DIR}/unit/Llama31TTServingTest.cpp")
  endif()
  set(_target "buddy-server-llama31-tt-${_test}-test")
  add_executable(${_target} "${_source}"
    "${_TT_MODEL_DIR}/Llama31TTExecution.cpp"
    "${_TT_MODEL_DIR}/Llama31TTResidentModel.cpp")
  target_compile_features(${_target} PRIVATE cxx_std_20)
  target_include_directories(${_target} PRIVATE "${_TT_MODEL_DIR}/include")
  target_link_libraries(${_target} PRIVATE
    buddy_runtime_core buddy_runtime_llm LLVMSupport)
  target_compile_definitions(${_target} PRIVATE
    BUDDY_LLAMA31_TT_TEST_RAX="${_TT_TEST_DIR}/legacy.rax"
    BUDDY_LLAMA31_TT_BATCH_TEST_RAX="${_TT_TEST_DIR}/batch.rax")
  add_dependencies(${_target} buddy-server-llama31-tt-test-fixtures)
  add_test(NAME buddy-server-llama31-tt-${_test}
    COMMAND ${_target} "${_TT_TEST_DIR}/serving.rax")
  set_tests_properties(buddy-server-llama31-tt-${_test} PROPERTIES TIMEOUT 30)
  list(APPEND BUDDY_SERVER_RESIDENT_ADAPTER_TARGETS ${_target})
endforeach()
