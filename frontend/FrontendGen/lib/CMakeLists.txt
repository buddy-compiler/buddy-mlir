include_directories()

antlr_target(FegenLexer FegenLexer.g4
            PACKAGE fegen
            LEXER
            )

antlr_target(FegenParser FegenParser.g4
            PACKAGE fegen
            DEPENDS_ANTLR FegenLexer
            PARSER
            LISTENER
            VISITOR
            COMPILE_FLAGS -lib 
            ${ANTLR_FegenLexer_OUTPUT_DIR}
            )

include_directories(${ANTLR_FegenLexer_OUTPUT_DIR}) 
set(ANTLR_FegenLexer_OUTPUT_DIR ${ANTLR_FegenLexer_OUTPUT_DIR} CACHE STRING "ANTLR_FegenLexer_OUTPUT_DIR")
include_directories(${ANTLR_FegenParser_OUTPUT_DIR})
set(ANTLR_FegenParser_OUTPUT_DIR ${ANTLR_FegenParser_OUTPUT_DIR} CACHE STRING "ANTLR_FegenParser_OUTPUT_DIR")

add_library(fegen_antlr_generated 
    ${ANTLR_FegenLexer_CXX_OUTPUTS} 
    ${ANTLR_FegenParser_CXX_OUTPUTS}
)

add_library(FegenRule
    FegenRule.cpp
)

llvm_map_components_to_libnames(llvm_libs support)

target_link_libraries(FegenRule
    PRIVATE
    ${llvm_libs}
)

add_library(fegenVisitor
    FegenVisitor.cpp
)

target_link_libraries(fegenVisitor
    PUBLIC
    fegen_antlr_generated
    antlr4_static
    FegenRule
)

