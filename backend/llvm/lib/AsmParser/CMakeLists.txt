# AsmParser

set(LLVM_AsmParser_DIR ${LLVM_MAIN_SRC_DIR}/lib/AsmParser)

add_llvm_component_library(LLVMBuddyAsmParser
  ${LLVM_AsmParser_DIR}/LLLexer.cpp
  ${LLVM_AsmParser_DIR}/LLParser.cpp
  ${LLVM_AsmParser_DIR}/Parser.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ASMParser

  DEPENDS
  buddy_intrinsics_gen

  LINK_COMPONENTS
  BinaryFormat
  BuddyCore
  Support
  )
