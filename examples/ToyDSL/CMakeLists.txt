# Necessary configuration for Antlr.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Add toy definition subdirectory.
add_subdirectory(include)
set(LLVM_LINK_COMPONENTS
  Support
)

# Call macro to add lexer and grammar to your build dependencies.
antlr_target(SampleGrammarLexer TLexer.g4 LEXER
             PACKAGE antlrcpptest)
antlr_target(SampleGrammarParser TParser.g4 PARSER
             PACKAGE antlrcpptest
             DEPENDS_ANTLR SampleGrammarLexer
             COMPILE_FLAGS -lib ${ANTLR_SampleGrammarLexer_OUTPUT_DIR})

# Include generated files in project environment
include_directories(${ANTLR_SampleGrammarLexer_OUTPUT_DIR})
include_directories(${ANTLR_SampleGrammarParser_OUTPUT_DIR})

# Include toy definition directories. 
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

# Add DSL -> MLIR frontend.
add_llvm_executable(buddy-toy-dsl main.cpp mlir/Dialect.cpp
                    ${ANTLR_SampleGrammarLexer_CXX_OUTPUTS}
                    ${ANTLR_SampleGrammarParser_CXX_OUTPUTS}
                    DEPENDS
                    ToyDSLOpsIncGen)

llvm_update_compile_flags(buddy-toy-dsl)
target_link_libraries(buddy-toy-dsl PRIVATE
  Threads::Threads
  antlr4_static
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
)
