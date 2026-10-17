# Test raw Qwen bytes and segmentation against the repository vocabulary.
add_executable(buddy-server-qwen3-vl-raw-codec-test
  "${CMAKE_CURRENT_LIST_DIR}/Qwen3VLRawCodecTest.cpp"
  "${BUDDY_SOURCE_DIR}/models/qwen3_vl/Qwen3VLRawCodec.cpp")
target_compile_features(buddy-server-qwen3-vl-raw-codec-test PRIVATE cxx_std_17)
target_include_directories(buddy-server-qwen3-vl-raw-codec-test PRIVATE
  "${BUDDY_SOURCE_DIR}/models/qwen3_vl/include")
target_link_libraries(buddy-server-qwen3-vl-raw-codec-test PRIVATE
  buddy_runtime_core buddy_runtime_llm)
add_test(NAME buddy-server-qwen3-vl-raw-codec
  COMMAND buddy-server-qwen3-vl-raw-codec-test
    "${BUDDY_SOURCE_DIR}/examples/BuddyQwen3/vocab.txt"
    "${CMAKE_CURRENT_BINARY_DIR}")
set_tests_properties(buddy-server-qwen3-vl-raw-codec PROPERTIES TIMEOUT 60)
list(APPEND BUDDY_SERVER_RESIDENT_ADAPTER_TARGETS
  buddy-server-qwen3-vl-raw-codec-test)
