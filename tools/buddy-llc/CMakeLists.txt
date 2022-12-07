set(LLVM_LINK_COMPONENTS
  BuddyRISCVCodeGen
  BuddyRISCVAsmParser
  BuddyRISCVDesc
  BuddyRISCVInfo

  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  
  Analysis
  AsmParser
  AsmPrinter
  CodeGen
  Core
  IRReader
  MC
  MIRParser
  Remarks
  ScalarOpts
  SelectionDAG
  Support
  Target
  TransformUtils
  Vectorize
  )

add_llvm_tool(buddy-llc
  ${LLVM_PROJECT_SOURCE_DIR}/llvm/tools/llc/llc.cpp

  DEPENDS
  buddy_intrinsics_gen
  SUPPORT_PLUGINS
  )

export_executable_symbols_for_plugins(buddy-llc)
