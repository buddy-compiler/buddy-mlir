# tools/buddy-server/CMakeLists.txt

add_executable(buddy-server
  buddy-server.cpp
  JsonCodec.cpp
  ResidentModelPluginHandle.cpp
  EmbeddingModelPluginHandle.cpp
  SimpleHttpServer.cpp
)

target_compile_features(buddy-server PRIVATE cxx_std_17)

target_include_directories(buddy-server PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(buddy-server PRIVATE
  buddy_runtime_core
  buddy_runtime_llm
  LLVMSupport
  ${CMAKE_DL_LIBS}
)

target_link_directories(buddy-server PRIVATE ${LLVM_LIBRARY_DIR})

if(BUDDY_MLIR_USE_MIMALLOC)
  target_link_libraries(buddy-server PRIVATE mimalloc)
endif()

install(TARGETS buddy-server
        EXPORT BuddyMLIRTargets
        COMPONENT buddy_tools)

if(BUDDY_ENABLE_TESTS)
  add_test(NAME buddy-server-help
    COMMAND buddy-server --help
  )
  add_executable(buddy-server-json-codec-test
    JsonCodecTest.cpp
    JsonCodec.cpp
  )
  target_compile_features(buddy-server-json-codec-test PRIVATE cxx_std_17)
  target_include_directories(buddy-server-json-codec-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(buddy-server-json-codec-test PRIVATE
    buddy_runtime_core
    ${CMAKE_DL_LIBS}
    buddy_runtime_llm
    LLVMSupport
  )
  target_link_directories(buddy-server-json-codec-test PRIVATE ${LLVM_LIBRARY_DIR})
  add_test(NAME buddy-server-json-codec
    COMMAND buddy-server-json-codec-test
  )
  set_tests_properties(buddy-server-json-codec PROPERTIES
    PASS_REGULAR_EXPRESSION "JsonCodec tests passed"
  )

  set_tests_properties(buddy-server-help PROPERTIES
    PASS_REGULAR_EXPRESSION "Usage:.*buddy-server"
  )

  add_custom_target(check-buddy-server
    COMMAND $<TARGET_FILE:buddy-server> --help
    DEPENDS buddy-server
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()

if(BUDDY_ENABLE_TESTS)
  add_executable(buddy-server-embedding-plugin-handle-test
    EmbeddingModelPluginHandleTest.cpp
    EmbeddingModelPluginHandle.cpp
  )
  target_compile_features(buddy-server-embedding-plugin-handle-test PRIVATE cxx_std_17)
  target_include_directories(buddy-server-embedding-plugin-handle-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(buddy-server-embedding-plugin-handle-test PRIVATE
    buddy_runtime_core
    ${CMAKE_DL_LIBS}
  )
  add_test(NAME buddy-server-embedding-plugin-handle
    COMMAND buddy-server-embedding-plugin-handle-test
  )
  set_tests_properties(buddy-server-embedding-plugin-handle PROPERTIES
    PASS_REGULAR_EXPRESSION "EmbeddingModelPluginHandle tests passed"
  )
endif()
