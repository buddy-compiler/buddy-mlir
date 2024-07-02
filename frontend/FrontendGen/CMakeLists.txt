add_subdirectory(lib)
set (LLVM_LINK_COMPONENTS 
    support
)

include_directories(${ANTLR_FegenLexer_OUTPUT_DIR})
include_directories(${ANTLR_FegenParser_OUTPUT_DIR})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_llvm_tool(buddy-frontendgen  
    frontendgen.cpp
)

target_link_libraries(buddy-frontendgen
    PRIVATE
    fegen_antlr_generated
    fegenVisitor
    antlr4_static
)