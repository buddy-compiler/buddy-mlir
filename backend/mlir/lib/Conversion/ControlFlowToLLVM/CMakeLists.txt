set(MLIR_CONVERSION_CONTROLFLOW_DIR ${LLVM_PROJECT_SOURCE_DIR}/mlir/lib/Conversion/ControlFlowToLLVM)

add_mlir_conversion_library(MLIRBuddyControlFlowToLLVM
  ${MLIR_CONVERSION_CONTROLFLOW_DIR}/ControlFlowToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ControlFlowToLLVM

  DEPENDS
  buddy_intrinsics_gen

  LINK_COMPONENTS
  BuddyCore

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRControlFlowDialect
  MLIRLLVMCommonConversion
  MLIRBuddyLLVMDialect
  MLIRPass
  MLIRTransformUtils
  )
