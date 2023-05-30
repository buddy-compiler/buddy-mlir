set(LLVM_LINK_COMPONENTS
  BuddyRISCVCodeGen
  BuddyRISCVAsmParser
  BuddyRISCVDesc
  BuddyRISCVInfo

  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  
  BuddyAnalysis
  BuddyAsmParser
  BuddyAsmPrinter
  BuddyCodeGen
  BuddyCore
  BuddyIRReader
  MC
  BuddyMIRParser
  BuddyRemarks
  BuddyScalarOpts
  BuddySelectionDAG
  Support
  BuddyTarget
  BuddyIPO
  BuddyVectorize
  TargetParser
  BuddyTransformUtils
)

add_llvm_tool(buddy-llc
  ${LLVM_PROJECT_SOURCE_DIR}/llvm/tools/llc/llc.cpp

  DEPENDS
  buddy_intrinsics_gen
  SUPPORT_PLUGINS
  )

export_executable_symbols_for_plugins(buddy-llc)
