set(MLIR_TARGET_LLVMIR_DIR ${LLVM_PROJECT_SOURCE_DIR}/mlir/lib/Target/LLVMIR)

add_mlir_translation_library(MLIRBuddyTargetLLVMIRExport
  ${MLIR_TARGET_LLVMIR_DIR}/DebugTranslation.cpp
  ${MLIR_TARGET_LLVMIR_DIR}/ModuleTranslation.cpp
  ${MLIR_TARGET_LLVMIR_DIR}/TypeToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  DEPENDS
  buddy_intrinsics_gen

  LINK_COMPONENTS
  BuddyCore
  FrontendOpenMP
  TransformUtils

  LINK_LIBS PUBLIC
  MLIRDLTIDialect
  MLIRBuddyLLVMDialect
  MLIRLLVMIRTransforms
  MLIRTranslateLib
)
