set(MLIR_DIALECT_LLVMIR_DIR ${LLVM_PROJECT_SOURCE_DIR}/mlir/lib/Dialect/LLVMIR)

add_mlir_dialect_library(MLIRBuddyLLVMDialect
  ${MLIR_DIALECT_LLVMIR_DIR}/IR/FunctionCallUtils.cpp
  ${MLIR_DIALECT_LLVMIR_DIR}/IR/LLVMAttrs.cpp
  ${MLIR_DIALECT_LLVMIR_DIR}/IR/LLVMDialect.cpp
  ${MLIR_DIALECT_LLVMIR_DIR}/IR/LLVMTypes.cpp
  ${MLIR_DIALECT_LLVMIR_DIR}/IR/LLVMTypeSyntax.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  DEPENDS
  buddy_intrinsics_gen

  LINK_COMPONENTS
  BuddyAsmParser
  BinaryFormat
  BitReader
  BitWriter
  Core

  LINK_LIBS PUBLIC
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
  )

add_mlir_dialect_library(MLIRBuddyNVVMDialect
  ${MLIR_DIALECT_LLVMIR_DIR}/IR/NVVMDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  DEPENDS
  buddy_intrinsics_gen

  LINK_COMPONENTS
  BuddyAsmParser
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBuddyLLVMDialect
  MLIRSideEffectInterfaces
  )

add_mlir_dialect_library(MLIRBuddyROCDLDialect
  ${MLIR_DIALECT_LLVMIR_DIR}/IR/ROCDLDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  DEPENDS
  buddy_intrinsics_gen

  LINK_COMPONENTS
  BuddyAsmParser
  BuddyCore

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRVectorToLLVM
  )
