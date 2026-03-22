add_executable(rax-inspect rax-inspect.cpp)

target_compile_features(rax-inspect PRIVATE cxx_std_17)

target_link_libraries(rax-inspect PRIVATE buddy_rax)

set_target_properties(rax-inspect PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
