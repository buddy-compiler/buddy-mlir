set(LLVM_Writer_DIR ${LLVM_MAIN_SRC_DIR}/lib/Bitcode/Writer)


add_llvm_component_library(LLVMBuddyBitWriter
  ${LLVM_Writer_DIR}/BitWriter.cpp
  ${LLVM_Writer_DIR}/BitcodeWriter.cpp
  ${LLVM_Writer_DIR}/BitcodeWriterPass.cpp
  ${LLVM_Writer_DIR}/ValueEnumerator.cpp

  DEPENDS
  buddy_intrinsics_gen

  LINK_COMPONENTS
  Analysis
  BuddyCore
  MC
  BuddyObject
  Object
  Support
  TargetParser
  )
