add_subdirectory(lib)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
link_directories("${CMAKE_CURRENT_BINARY_DIR}/lib")