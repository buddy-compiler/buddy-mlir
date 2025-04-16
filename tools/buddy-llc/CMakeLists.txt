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
  Passes
)

add_llvm_tool(buddy-llc
  ${LLVM_MAIN_SRC_DIR}/tools/llc/llc.cpp
  ${LLVM_MAIN_SRC_DIR}/tools/llc/NewPMDriver.cpp

  DEPENDS
  buddy_intrinsics_gen
  SUPPORT_PLUGINS
  )

export_executable_symbols_for_plugins(buddy-llc)
