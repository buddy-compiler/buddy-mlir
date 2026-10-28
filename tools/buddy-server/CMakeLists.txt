# tools/buddy-server/CMakeLists.txt

find_package(Threads REQUIRED)

add_library(buddy-server-core STATIC
  src/BuddyServerApp.cpp
  src/http/SimpleHttpServer.cpp
  src/plugin/AudioTranscriptionModelPluginHandle.cpp
  src/plugin/EmbeddingModelPluginHandle.cpp
  src/plugin/MaskedLMModelPluginHandle.cpp
  src/plugin/ModelPluginDiscovery.cpp
  src/plugin/ResidentModelPluginHandle.cpp
  src/protocol/AudioTranscriptionJsonCodec.cpp
  src/protocol/JsonCodec.cpp
)
target_compile_features(buddy-server-core PUBLIC cxx_std_17)
target_include_directories(buddy-server-core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/http
  ${CMAKE_CURRENT_SOURCE_DIR}/src/plugin
  ${CMAKE_CURRENT_SOURCE_DIR}/src/protocol
)
target_link_libraries(buddy-server-core PUBLIC
  buddy_runtime_core
  buddy_runtime_llm
  buddy_runtime_rax_dependencies
  LLVMSupport
  Threads::Threads
  ${CMAKE_DL_LIBS}
)
target_link_directories(buddy-server-core PUBLIC ${LLVM_LIBRARY_DIR})

add_executable(buddy-server src/buddy-server.cpp)
target_compile_features(buddy-server PRIVATE cxx_std_17)
target_compile_definitions(buddy-server PRIVATE
  BUDDY_VERSION="${BUDDY_PACKAGE_VERSION}")
target_link_libraries(buddy-server PRIVATE
  buddy-server-core
  CLI11
)
if(BUDDY_MLIR_USE_MIMALLOC)
  target_link_libraries(buddy-server PRIVATE mimalloc)
endif()

install(TARGETS buddy-server
  EXPORT BuddyMLIRTargets
  COMPONENT buddy_tools)

if(BUDDY_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
