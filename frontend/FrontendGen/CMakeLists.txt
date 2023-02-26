include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
link_directories("${CMAKE_CURRENT_BINARY_DIR}/lib")
add_subdirectory(lib)
set (LLVM_LINK_COMPONENTS 
support
frontendgenlib
) 
add_llvm_tool(buddy-frontendgen  
frontendgen.cpp
)
