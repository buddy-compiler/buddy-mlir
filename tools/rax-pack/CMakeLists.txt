add_executable(rax-pack rax-pack.cpp)

target_compile_features(rax-pack PRIVATE cxx_std_17)

target_link_libraries(rax-pack PRIVATE
  buddy_rax
  BuddyRHAL
  MLIRParser
  MLIRIR
  MLIRSupport
)

# Include midend generated headers (RHALOps*.inc)
target_include_directories(rax-pack PRIVATE
  ${BUDDY_SOURCE_DIR}/midend/include/Dialect
  ${CMAKE_BINARY_DIR}/midend/include/Dialect
)

set_target_properties(rax-pack PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
