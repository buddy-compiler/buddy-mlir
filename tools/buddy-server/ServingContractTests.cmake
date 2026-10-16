# Execute the shared generation controller through real HTTP sockets. Only
# execution and raw fixture vocabulary are synthetic; this is not inference.
set(_CONTRACT_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/serving-contract-test")
set(_CONTRACT_RAX_FILES)
foreach(_profile resident embedding)
  set(_plugin "buddy-server-contract-${_profile}-plugin")
  add_library(${_plugin} MODULE ServingContractFakePlugin.cpp)
  target_compile_features(${_plugin} PRIVATE cxx_std_17)
  target_link_libraries(${_plugin} PRIVATE buddy_runtime_core buddy_runtime_llm)
  if(_profile STREQUAL "embedding")
    target_compile_definitions(${_plugin} PRIVATE BUDDY_CONTRACT_EMBEDDING=1)
  endif()
  set(_rax "${_CONTRACT_FIXTURE_DIR}/${_profile}/service.rax")
  add_custom_command(OUTPUT "${_rax}"
    COMMAND ${CMAKE_COMMAND} -E make_directory
      "${_CONTRACT_FIXTURE_DIR}/${_profile}"
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${_plugin}>"
      "${_CONTRACT_FIXTURE_DIR}/${_profile}/service.so"
    COMMAND $<TARGET_FILE:rax-pack>
      "${CMAKE_CURRENT_SOURCE_DIR}/ServingContractManifestTest.mlir" -o "${_rax}"
    DEPENDS rax-pack ${_plugin}
      "${CMAKE_CURRENT_SOURCE_DIR}/ServingContractManifestTest.mlir"
    VERBATIM)
  list(APPEND _CONTRACT_RAX_FILES "${_rax}")
endforeach()
add_custom_target(buddy-server-serving-contract-fixtures
  DEPENDS ${_CONTRACT_RAX_FILES})
add_test(NAME buddy-server-serving-http-contract
  COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/BuddyServerServingIntegrationTest.py"
    "$<TARGET_FILE:buddy-server>"
    "${_CONTRACT_FIXTURE_DIR}/resident/service.rax"
    "${_CONTRACT_FIXTURE_DIR}/embedding/service.rax"
    "${CMAKE_CURRENT_SOURCE_DIR}/compatibility/target-contract.json"
    "${_CONTRACT_FIXTURE_DIR}/scratch")
set_tests_properties(buddy-server-serving-http-contract PROPERTIES TIMEOUT 90)
