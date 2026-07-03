# tools/buddy-server/CMakeLists.txt

add_executable(buddy-server
  buddy-server.cpp
  JsonCodec.cpp
  ResidentModelPluginHandle.cpp
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
  set_tests_properties(buddy-server-help PROPERTIES
    PASS_REGULAR_EXPRESSION "Usage:.*buddy-server"
  )

  add_custom_target(check-buddy-server
    COMMAND $<TARGET_FILE:buddy-server> --help
    DEPENDS buddy-server
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()
