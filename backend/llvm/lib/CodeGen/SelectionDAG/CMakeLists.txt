set(LLVM_SelectionDAG_DIR ${LLVM_MAIN_SRC_DIR}/lib/CodeGen/SelectionDAG)

add_llvm_component_library(LLVMBuddySelectionDAG
  ${LLVM_SelectionDAG_DIR}/DAGCombiner.cpp
  ${LLVM_SelectionDAG_DIR}/FastISel.cpp
  ${LLVM_SelectionDAG_DIR}/FunctionLoweringInfo.cpp
  ${LLVM_SelectionDAG_DIR}/InstrEmitter.cpp
  ${LLVM_SelectionDAG_DIR}/LegalizeDAG.cpp
  ${LLVM_SelectionDAG_DIR}/LegalizeFloatTypes.cpp
  ${LLVM_SelectionDAG_DIR}/LegalizeIntegerTypes.cpp
  ${LLVM_SelectionDAG_DIR}/LegalizeTypes.cpp
  ${LLVM_SelectionDAG_DIR}/LegalizeTypesGeneric.cpp
  ${LLVM_SelectionDAG_DIR}/LegalizeVectorOps.cpp
  ${LLVM_SelectionDAG_DIR}/LegalizeVectorTypes.cpp
  ${LLVM_SelectionDAG_DIR}/ResourcePriorityQueue.cpp
  ${LLVM_SelectionDAG_DIR}/ScheduleDAGFast.cpp
  ${LLVM_SelectionDAG_DIR}/ScheduleDAGRRList.cpp
  ${LLVM_SelectionDAG_DIR}/ScheduleDAGSDNodes.cpp
  ${LLVM_SelectionDAG_DIR}/ScheduleDAGVLIW.cpp
  ${LLVM_SelectionDAG_DIR}/SelectionDAGBuilder.cpp
  ${LLVM_SelectionDAG_DIR}/SelectionDAG.cpp
  ${LLVM_SelectionDAG_DIR}/SelectionDAGAddressAnalysis.cpp
  ${LLVM_SelectionDAG_DIR}/SelectionDAGDumper.cpp
  ${LLVM_SelectionDAG_DIR}/SelectionDAGISel.cpp
  ${LLVM_SelectionDAG_DIR}/SelectionDAGPrinter.cpp
  ${LLVM_SelectionDAG_DIR}/SelectionDAGTargetInfo.cpp
  ${LLVM_SelectionDAG_DIR}/StatepointLowering.cpp
  ${LLVM_SelectionDAG_DIR}/TargetLowering.cpp

  DEPENDS
  buddy_intrinsics_gen

  LINK_COMPONENTS
  BuddyAnalysis
  BuddyCodeGen
  BuddyCore
  MC
  Support
  BuddyTarget
  BuddyTransformUtils
  )
