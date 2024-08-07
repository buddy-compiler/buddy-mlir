set(LLVM_MIRParser_DIR ${LLVM_MAIN_SRC_DIR}/lib/CodeGen/MIRParser)

add_llvm_component_library(LLVMBuddyMIRParser
  ${LLVM_MIRParser_DIR}/MILexer.cpp
  ${LLVM_MIRParser_DIR}/MIParser.cpp
  ${LLVM_MIRParser_DIR}/MIRParser.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/CodeGen/MIRParser

  DEPENDS
  buddy_intrinsics_gen

  LINK_COMPONENTS
  BuddyAsmParser
  BinaryFormat
  BuddyCodeGen
  BuddyCore
  MC
  Support
  BuddyTarget
  )
