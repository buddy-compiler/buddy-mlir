list(APPEND LLVM_COMMON_DEPENDS buddy_intrinsics_gen)

list(APPEND LLVM_TABLEGEN_FLAGS -I ${LLVM_MAIN_SRC_DIR}/lib/Target)

set(LLVM_Target_DIR ${LLVM_MAIN_SRC_DIR}/lib/Target)

add_llvm_component_library(LLVMBuddyTarget
  ${LLVM_Target_DIR}/Target.cpp
  ${LLVM_Target_DIR}/TargetIntrinsicInfo.cpp
  ${LLVM_Target_DIR}/TargetLoweringObjectFile.cpp
  ${LLVM_Target_DIR}/TargetMachine.cpp
  ${LLVM_Target_DIR}/TargetMachineC.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Target

  LINK_COMPONENTS
  BuddyAnalysis
  Core
  MC
  Support
  )

add_subdirectory(RISCV)
