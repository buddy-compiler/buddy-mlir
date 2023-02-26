# ------------------------------------------------------------------------------
# Helper functions and variables.
# ------------------------------------------------------------------------------
function(copy_file_to_build_dir src_dir output_dir file)
  set(src ${src_dir}/${file})
  set(dst ${output_dir}/${file})
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying RISCV Target - ${file}...")
endfunction(copy_file_to_build_dir)

set(LLVM_TARGET_RISCV_DIR ${LLVM_PROJECT_SOURCE_DIR}/llvm/lib/Target/RISCV)

# ------------------------------------------------------------------------------
# Configure RISC-V Buddy Extension.
# ------------------------------------------------------------------------------

add_llvm_component_group(RISCVBuddyExt)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${BUDDY_SOURCE_DIR}/backend/include/)
include_directories(${LLVM_TARGET_RISCV_DIR})

set(LLVM_TARGET_DEFINITIONS RISCVBuddyExt.td)

tablegen(LLVM RISCVGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM RISCVGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM RISCVGenCompressInstEmitter.inc -gen-compress-inst-emitter)
tablegen(LLVM RISCVGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM RISCVGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM RISCVGenGlobalISel.inc -gen-global-isel)
tablegen(LLVM RISCVGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM RISCVGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM RISCVGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM RISCVGenRegisterBank.inc -gen-register-bank)
tablegen(LLVM RISCVGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM RISCVGenSearchableTables.inc -gen-searchable-tables)
tablegen(LLVM RISCVGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(RISCVCommonTableGen)

# ------------------------------------------------------------------------------
# Build RISC-V Buddy Extension CodeGen.
# ------------------------------------------------------------------------------

# Copy related files to build directory.
file(GLOB UPSTREAM_LLVM_TARGET_RISCV_FILE
  RELATIVE ${LLVM_TARGET_RISCV_DIR}
  "${LLVM_TARGET_RISCV_DIR}/*.cpp"
  "${LLVM_TARGET_RISCV_DIR}/*.h"
)
foreach(file ${UPSTREAM_LLVM_TARGET_RISCV_FILE})
  copy_file_to_build_dir(${LLVM_TARGET_RISCV_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${file})
endforeach()

# Copy related files to build directory.
file(GLOB UPSTREAM_LLVM_TARGET_RISCV_GISEL_FILE
  RELATIVE ${LLVM_TARGET_RISCV_DIR}/GISel/
  "${LLVM_TARGET_RISCV_DIR}/GISel/*.cpp"
  "${LLVM_TARGET_RISCV_DIR}/GISel/*.h"
)
foreach(file ${UPSTREAM_LLVM_TARGET_RISCV_GISEL_FILE})
  copy_file_to_build_dir(${LLVM_TARGET_RISCV_DIR}/GISel/ ${CMAKE_CURRENT_BINARY_DIR} ${file})
endforeach()

# Build BuddyRISCVCodeGen target.
add_llvm_target(BuddyRISCVCodeGen
  # Target RISCV
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVAsmPrinter.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVCodeGenPrepare.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMakeCompressible.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVExpandAtomicPseudoInsts.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVExpandPseudoInsts.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVFrameLowering.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVGatherScatterLowering.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVInsertVSETVLI.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVInstrInfo.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVISelDAGToDAG.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVISelLowering.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMachineFunctionInfo.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMacroFusion.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMCInstLower.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMergeBaseOffset.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVRedundantCopyElimination.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVRegisterInfo.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVSExtWRemoval.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVSubtarget.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVTargetMachine.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVTargetObjectFile.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVTargetTransformInfo.cpp
  # GISel
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVCallLowering.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVInstructionSelector.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVLegalizerInfo.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVRegisterBankInfo.cpp
  # Add *.h files to track the copies above.
  # Target RISCV
  ${CMAKE_CURRENT_BINARY_DIR}/RISCV.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVFrameLowering.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVInstrInfo.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVISelDAGToDAG.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVISelLowering.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMachineFunctionInfo.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMacroFusion.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVRegisterInfo.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVSubtarget.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVTargetMachine.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVTargetObjectFile.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVTargetTransformInfo.h
  # GISel
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVCallLowering.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVLegalizerInfo.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVRegisterBankInfo.h

  LINK_COMPONENTS
  BuddyAnalysis
  BuddyAsmPrinter
  BuddyCore
  BuddyIPO
  BuddyCodeGen
  MC
  RISCVDesc
  RISCVInfo
  BuddySelectionDAG
  Support
  BuddyTarget
  TransformUtils
  BuddyGlobalISel

  ADD_TO_COMPONENT
  RISCVBuddyExt
  )

# ------------------------------------------------------------------------------
# Build RISC-V Buddy Extension Asm Parser.
# ------------------------------------------------------------------------------

# Copy related files to build directory.
file(GLOB UPSTREAM_LLVM_TARGET_RISCV_ASMPARSER_FILE
  RELATIVE ${LLVM_TARGET_RISCV_DIR}/AsmParser/
  "${LLVM_TARGET_RISCV_DIR}/AsmParser/*.cpp"
)
foreach(file ${UPSTREAM_LLVM_TARGET_RISCV_ASMPARSER_FILE})
  copy_file_to_build_dir(${LLVM_TARGET_RISCV_DIR}/AsmParser ${CMAKE_CURRENT_BINARY_DIR} ${file})
endforeach()

# Build LLVMBuddyRISCVAsmParser target.
add_llvm_component_library(LLVMBuddyRISCVAsmParser
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVAsmParser.cpp
  LINK_COMPONENTS
  MC
  MCParser
  RISCVDesc
  RISCVInfo
  Support

  ADD_TO_COMPONENT
  RISCVBuddyExt
  )

# ------------------------------------------------------------------------------
# Build RISC-V Buddy Extension Desc.
# ------------------------------------------------------------------------------

# Copy related files to build directory.
file(GLOB UPSTREAM_LLVM_TARGET_RISCV_DESC_FILE
  RELATIVE ${LLVM_TARGET_RISCV_DIR}/MCTargetDesc/
  "${LLVM_TARGET_RISCV_DIR}/MCTargetDesc/*.cpp"
  "${LLVM_TARGET_RISCV_DIR}/MCTargetDesc/*.h"
)
foreach(file ${UPSTREAM_LLVM_TARGET_RISCV_DESC_FILE})
  copy_file_to_build_dir(${LLVM_TARGET_RISCV_DIR}/MCTargetDesc ${CMAKE_CURRENT_BINARY_DIR} ${file})
endforeach()

# Build LLVMBuddyRISCVDesc target.
add_llvm_component_library(LLVMBuddyRISCVDesc
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVAsmBackend.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVBaseInfo.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVELFObjectWriter.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVInstPrinter.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMCAsmInfo.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMCCodeEmitter.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMCExpr.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMCObjectFileInfo.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMCTargetDesc.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMatInt.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVTargetStreamer.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVELFStreamer.cpp
  # Add *.h files to track the copies above.
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVAsmBackend.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVBaseInfo.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVELFStreamer.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVFixupKinds.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVInstPrinter.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMatInt.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMCAsmInfo.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMCExpr.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMCObjectFileInfo.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMCTargetDesc.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVTargetStreamer.h

  LINK_COMPONENTS
  MC
  RISCVInfo
  Support

  ADD_TO_COMPONENT
  RISCVBuddyExt
)

# ------------------------------------------------------------------------------
# Build RISC-V Buddy Extension Info.
# ------------------------------------------------------------------------------

# Copy related files to build directory.
file(GLOB UPSTREAM_LLVM_TARGET_RISCV_TARGETINFO_FILE
  RELATIVE ${LLVM_TARGET_RISCV_DIR}/TargetInfo/
  "${LLVM_TARGET_RISCV_DIR}/TargetInfo/*.cpp"
  "${LLVM_TARGET_RISCV_DIR}/TargetInfo/*.h"
)
foreach(file ${UPSTREAM_LLVM_TARGET_RISCV_TARGETINFO_FILE})
  copy_file_to_build_dir(${LLVM_TARGET_RISCV_DIR}/TargetInfo ${CMAKE_CURRENT_BINARY_DIR} ${file})
endforeach()

# Build LLVMBuddyRISCVInfo target.
add_llvm_component_library(LLVMBuddyRISCVInfo
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVTargetInfo.cpp
  # Add *.h files to track the copies above.
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVTargetInfo.h

  LINK_COMPONENTS
  MC
  Support

  ADD_TO_COMPONENT
  RISCVBuddyExt
  )
