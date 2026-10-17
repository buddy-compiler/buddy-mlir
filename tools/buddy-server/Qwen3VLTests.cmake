# Execute production Qwen runtime/resident code through actual dlopen kernels.
# Resources and kernel arithmetic are synthetic; this does not run a model.
set(_QWEN_MODEL_DIR "${BUDDY_SOURCE_DIR}/models/qwen3_vl")
set(_QWEN_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/qwen3-vl-test")
set(_QWEN_GENERATOR
  "${CMAKE_CURRENT_LIST_DIR}/testdata/generate_qwen3_vl_fixture.py")
add_library(buddy-server-qwen3-vl-test-kernels MODULE
  "${CMAKE_CURRENT_LIST_DIR}/Qwen3VLTestKernels.cpp")
target_compile_features(buddy-server-qwen3-vl-test-kernels PRIVATE cxx_std_17)
set(_QWEN_TEST_RAX_FILES)
foreach(_profile canonical shifted length full near)
  list(APPEND _QWEN_TEST_RAX_FILES "${_QWEN_TEST_DIR}/${_profile}.rax")
endforeach()
add_custom_command(OUTPUT ${_QWEN_TEST_RAX_FILES}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_QWEN_TEST_DIR}"
  COMMAND "${Python3_EXECUTABLE}" "${_QWEN_GENERATOR}"
    --output-dir "${_QWEN_TEST_DIR}"
    --kernels "$<TARGET_FILE:buddy-server-qwen3-vl-test-kernels>"
  COMMAND $<TARGET_FILE:rax-pack> "${_QWEN_TEST_DIR}/canonical.mlir"
    -o "${_QWEN_TEST_DIR}/canonical.rax"
  COMMAND $<TARGET_FILE:rax-pack> "${_QWEN_TEST_DIR}/shifted.mlir"
    -o "${_QWEN_TEST_DIR}/shifted.rax"
  COMMAND $<TARGET_FILE:rax-pack> "${_QWEN_TEST_DIR}/length.mlir"
    -o "${_QWEN_TEST_DIR}/length.rax"
  COMMAND $<TARGET_FILE:rax-pack> "${_QWEN_TEST_DIR}/full.mlir"
    -o "${_QWEN_TEST_DIR}/full.rax"
  COMMAND $<TARGET_FILE:rax-pack> "${_QWEN_TEST_DIR}/near.mlir"
    -o "${_QWEN_TEST_DIR}/near.rax"
  DEPENDS rax-pack buddy-server-qwen3-vl-test-kernels "${_QWEN_GENERATOR}"
  VERBATIM)
add_custom_target(buddy-server-qwen3-vl-test-fixtures
  DEPENDS ${_QWEN_TEST_RAX_FILES})

add_executable(buddy-server-qwen3-vl-serving-test
  "${CMAKE_CURRENT_LIST_DIR}/Qwen3VLServingTest.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/JsonCodec.cpp"
  "${_QWEN_MODEL_DIR}/Qwen3VLRawCodec.cpp"
  "${_QWEN_MODEL_DIR}/Qwen3VLRuntime.cpp"
  "${_QWEN_MODEL_DIR}/Qwen3VLResidentModel.cpp")
target_compile_features(buddy-server-qwen3-vl-serving-test PRIVATE cxx_std_17)
target_include_directories(buddy-server-qwen3-vl-serving-test PRIVATE
  "${_QWEN_MODEL_DIR}/include"
  "${BUDDY_SOURCE_DIR}/thirdparty/include")
target_link_libraries(buddy-server-qwen3-vl-serving-test PRIVATE
  buddy_runtime_core buddy_runtime_llm LLVMSupport ${CMAKE_DL_LIBS})
add_dependencies(buddy-server-qwen3-vl-serving-test
  buddy-server-qwen3-vl-test-fixtures)
add_test(NAME buddy-server-qwen3-vl-serving
  COMMAND buddy-server-qwen3-vl-serving-test "${_QWEN_TEST_DIR}"
    "$<TARGET_FILE:buddy-server-qwen3-vl-test-kernels>")
set_tests_properties(buddy-server-qwen3-vl-serving PROPERTIES TIMEOUT 60)

list(APPEND BUDDY_SERVER_RESIDENT_ADAPTER_TARGETS
  buddy-server-qwen3-vl-serving-test)
