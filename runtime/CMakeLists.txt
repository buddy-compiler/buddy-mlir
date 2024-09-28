add_subdirectory(lib)
if(BUDDY_MLIR_ENABLE_PYTHON_PACKAGES)
  add_subdirectory(Python)
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
link_directories("${CMAKE_CURRENT_BINARY_DIR}/lib")

add_llvm_tool(buddy-profiler
buddy-profiler.cpp
)

target_link_libraries(buddy-profiler
  PRIVATE
  BuddyRuntime
)

