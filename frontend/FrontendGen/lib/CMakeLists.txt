include_directories(../include)
set(LLVM_LINK_COMPONENTS
support)

add_llvm_component_library(LLVMfrontendgenlib
CGModule.cpp   
Lexer.cpp 
Parser.cpp 
Sema.cpp
Diagnostics.cpp
LINK_COMPONENTS
support)
