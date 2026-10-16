# Compile the production resident plugins without importing weights or building
# model kernels. Keep these checks available when all BUDDY_BUILD_*_MODEL
# options are OFF, so interface and link regressions reach the default tests.
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(_RESIDENT_MODELS_DIR "${BUDDY_SOURCE_DIR}/models")
set(_RESIDENT_CODEGEN_DIR "${BUDDY_SOURCE_DIR}/tools/buddy-codegen")
set(_RESIDENT_GENERATED_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/resident-adapters/deepseek/generated")
set(_RESIDENT_DEEPSEEK_CONFIG "${_RESIDENT_GENERATED_DIR}/config.json")
set(_RESIDENT_DEEPSEEK_HEADER
  "${_RESIDENT_GENERATED_DIR}/buddy/runtime/models/ModelSession.h")
set(_RESIDENT_DEEPSEEK_SESSION
  "${_RESIDENT_GENERATED_DIR}/ModelSession.cpp")
set(_RESIDENT_DEEPSEEK_HF_CONFIG
  "${CMAKE_CURRENT_LIST_DIR}/testdata/deepseek-r1-config.json")
set(_RESIDENT_DEEPSEEK_SPEC
  "${_RESIDENT_MODELS_DIR}/deepseek_r1/specs/f32.json")

add_custom_command(
  OUTPUT "${_RESIDENT_DEEPSEEK_CONFIG}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_RESIDENT_GENERATED_DIR}"
  COMMAND "${Python3_EXECUTABLE}" "${_RESIDENT_CODEGEN_DIR}/gen_config.py"
          --spec "${_RESIDENT_DEEPSEEK_SPEC}"
          --hf-config "${_RESIDENT_DEEPSEEK_HF_CONFIG}"
          -o "${_RESIDENT_DEEPSEEK_CONFIG}"
  DEPENDS
    "${_RESIDENT_DEEPSEEK_SPEC}"
    "${_RESIDENT_DEEPSEEK_HF_CONFIG}"
    "${_RESIDENT_CODEGEN_DIR}/gen_config.py"
  COMMENT "Generating offline DeepSeek resident check config"
  VERBATIM)
add_custom_command(
  OUTPUT "${_RESIDENT_DEEPSEEK_HEADER}" "${_RESIDENT_DEEPSEEK_SESSION}"
  COMMAND "${Python3_EXECUTABLE}" "${_RESIDENT_CODEGEN_DIR}/gen_session.py"
          --config "${_RESIDENT_DEEPSEEK_CONFIG}"
          --output-dir "${_RESIDENT_GENERATED_DIR}"
  DEPENDS "${_RESIDENT_DEEPSEEK_CONFIG}"
          "${_RESIDENT_CODEGEN_DIR}/gen_session.py"
  COMMENT "Generating production DeepSeek ModelSession for resident check"
  VERBATIM)
add_custom_target(buddy-server-deepseek-resident-generated
  DEPENDS "${_RESIDENT_DEEPSEEK_HEADER}" "${_RESIDENT_DEEPSEEK_SESSION}")

add_library(buddy-server-deepseek-resident-link-check MODULE
  "${_RESIDENT_MODELS_DIR}/deepseek_r1/DeepSeekR1RawCodec.cpp"
  "${_RESIDENT_MODELS_DIR}/deepseek_r1/DeepSeekR1ResidentModel.cpp"
  "${_RESIDENT_MODELS_DIR}/deepseek_r1/DeepSeekR1ResidentModelPlugin.cpp"
  "${_RESIDENT_DEEPSEEK_SESSION}")
add_dependencies(buddy-server-deepseek-resident-link-check
  buddy-server-deepseek-resident-generated)
target_include_directories(buddy-server-deepseek-resident-link-check PRIVATE
  "${_RESIDENT_MODELS_DIR}/deepseek_r1/include"
  "${_RESIDENT_GENERATED_DIR}")

add_library(buddy-server-qwen3-vl-resident-link-check MODULE
  "${_RESIDENT_MODELS_DIR}/qwen3_vl/Qwen3VLRuntime.cpp"
  "${_RESIDENT_MODELS_DIR}/qwen3_vl/Qwen3VLResidentModel.cpp"
  "${_RESIDENT_MODELS_DIR}/qwen3_vl/Qwen3VLResidentModelPlugin.cpp")
target_include_directories(buddy-server-qwen3-vl-resident-link-check PRIVATE
  "${_RESIDENT_MODELS_DIR}/qwen3_vl/include"
  "${BUDDY_SOURCE_DIR}/thirdparty/include")

add_library(buddy-server-llama31-tt-resident-portable-link-check MODULE
  "${_RESIDENT_MODELS_DIR}/llama31_tt/Llama31TTExecution.cpp"
  "${_RESIDENT_MODELS_DIR}/llama31_tt/Llama31TTResidentModel.cpp"
  "${_RESIDENT_MODELS_DIR}/llama31_tt/Llama31TTResidentModelPlugin.cpp"
  "${_RESIDENT_MODELS_DIR}/llama31_tt/Llama31TTNNBackend.cpp")
target_include_directories(
  buddy-server-llama31-tt-resident-portable-link-check PRIVATE
  "${_RESIDENT_MODELS_DIR}/llama31_tt/include")
target_compile_definitions(
  buddy-server-llama31-tt-resident-portable-link-check PRIVATE
  BUDDY_LLAMA31_TT_HAS_TTNN=0)
target_compile_features(
  buddy-server-llama31-tt-resident-portable-link-check PRIVATE cxx_std_20)

set(_RESIDENT_ADAPTER_PLUGINS
  buddy-server-deepseek-resident-link-check
  buddy-server-qwen3-vl-resident-link-check
  buddy-server-llama31-tt-resident-portable-link-check)
foreach(_plugin IN LISTS _RESIDENT_ADAPTER_PLUGINS)
  target_compile_features(${_plugin} PRIVATE cxx_std_17)
  target_link_libraries(${_plugin} PRIVATE
    buddy_runtime_core buddy_runtime_llm LLVMSupport ${CMAKE_DL_LIBS})
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(${_plugin} PRIVATE "LINKER:-z,defs")
  endif()
endforeach()

add_executable(buddy-server-resident-adapter-load-test
  "${CMAKE_CURRENT_LIST_DIR}/ResidentAdapterLoadTest.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ModelPluginDiscovery.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ResidentModelPluginHandle.cpp")
target_compile_features(buddy-server-resident-adapter-load-test PRIVATE cxx_std_17)
target_link_libraries(buddy-server-resident-adapter-load-test PRIVATE
  buddy_runtime_core ${CMAKE_DL_LIBS})
add_dependencies(buddy-server-resident-adapter-load-test
  ${_RESIDENT_ADAPTER_PLUGINS})

add_test(NAME buddy-server-resident-adapter-deepseek
  COMMAND buddy-server-resident-adapter-load-test
          "$<TARGET_FILE:buddy-server-deepseek-resident-link-check>"
          deepseek_r1)
add_test(NAME buddy-server-resident-adapter-qwen3-vl
  COMMAND buddy-server-resident-adapter-load-test
          "$<TARGET_FILE:buddy-server-qwen3-vl-resident-link-check>"
          qwen3_vl)
add_test(NAME buddy-server-resident-adapter-llama31-tt-portable
  COMMAND buddy-server-resident-adapter-load-test
          "$<TARGET_FILE:buddy-server-llama31-tt-resident-portable-link-check>"
          llama31_tt)
set_tests_properties(
  buddy-server-resident-adapter-deepseek
  buddy-server-resident-adapter-qwen3-vl
  buddy-server-resident-adapter-llama31-tt-portable
  PROPERTIES TIMEOUT 20)

set(BUDDY_SERVER_RESIDENT_ADAPTER_TARGETS
  buddy-server-resident-adapter-load-test ${_RESIDENT_ADAPTER_PLUGINS})
add_custom_target(check-buddy-resident-adapters
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j1
          -R "^buddy-server-resident-adapter-"
  DEPENDS ${BUDDY_SERVER_RESIDENT_ADAPTER_TARGETS}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# This executes the production adapter/raw codec with an injected LLMSession;
# the generated session is still compiled and linked, without loading weights.
add_executable(buddy-server-deepseek-serving-test
  "${CMAKE_CURRENT_LIST_DIR}/DeepSeekR1ServingTest.cpp"
  "${_RESIDENT_MODELS_DIR}/deepseek_r1/DeepSeekR1ResidentModel.cpp"
  "${_RESIDENT_MODELS_DIR}/deepseek_r1/DeepSeekR1RawCodec.cpp"
  "${_RESIDENT_DEEPSEEK_SESSION}")
add_dependencies(buddy-server-deepseek-serving-test
  buddy-server-deepseek-resident-generated)
target_compile_features(buddy-server-deepseek-serving-test PRIVATE cxx_std_17)
target_include_directories(buddy-server-deepseek-serving-test PRIVATE
  "${_RESIDENT_MODELS_DIR}/deepseek_r1/include"
  "${_RESIDENT_GENERATED_DIR}")
target_link_libraries(buddy-server-deepseek-serving-test PRIVATE
  buddy_runtime_core buddy_runtime_llm LLVMSupport ${CMAKE_DL_LIBS})
add_test(NAME buddy-server-deepseek-serving
  COMMAND buddy-server-deepseek-serving-test
    "${BUDDY_SOURCE_DIR}/examples/BuddyDeepSeekR1/vocab.txt")
set_tests_properties(buddy-server-deepseek-serving PROPERTIES TIMEOUT 30)
list(APPEND BUDDY_SERVER_RESIDENT_ADAPTER_TARGETS
  buddy-server-deepseek-serving-test)

include(${CMAKE_CURRENT_LIST_DIR}/Llama31TTTests.cmake)
