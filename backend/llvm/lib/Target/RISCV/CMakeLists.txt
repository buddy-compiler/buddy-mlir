# ------------------------------------------------------------------------------
# Helper functions macro and variables.
# ------------------------------------------------------------------------------
function(copy_file_to_build_dir src_dir output_dir file)
  set(src ${src_dir}/${file})
  set(dst ${output_dir}/${file})
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying RISCV Target - ${file}...")
endfunction(copy_file_to_build_dir)

# The macro helps to find the correct path of "IntrinsicsRISCV.h"
macro(buddy_add_llvm_target target_name)
  include_directories(BEFORE
    ${BUDDY_BUILD_DIR}/backend/include/
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
  add_llvm_component_library(LLVM${target_name} ${ARGN})
  set( CURRENT_LLVM_TARGET LLVM${target_name} )
endmacro(buddy_add_llvm_target)

set(LLVM_TARGET_RISCV_DIR ${LLVM_MAIN_SRC_DIR}/lib/Target/RISCV)

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
tablegen(LLVM RISCVGenMacroFusion.inc -gen-macro-fusion-pred)
tablegen(LLVM RISCVGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM RISCVGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM RISCVGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM RISCVGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM RISCVGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM RISCVGenRegisterBank.inc -gen-register-bank)
tablegen(LLVM RISCVGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM RISCVGenSearchableTables.inc -gen-searchable-tables)
tablegen(LLVM RISCVGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM RISCVGenExegesis.inc -gen-exegesis)

set(LLVM_TARGET_DEFINITIONS ${LLVM_TARGET_RISCV_DIR}/RISCVGISel.td)
tablegen(LLVM RISCVGenGlobalISel.inc -gen-global-isel)
tablegen(LLVM RISCVGenO0PreLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="RISCVO0PreLegalizerCombiner")
tablegen(LLVM RISCVGenPreLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="RISCVPreLegalizerCombiner")
tablegen(LLVM RISCVGenPostLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="RISCVPostLegalizerCombiner")

add_public_tablegen_target(RISCVBuddyCommonTableGen)

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
buddy_add_llvm_target(BuddyRISCVCodeGen
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVAsmPrinter.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVCallingConv.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVCodeGenPrepare.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVConstantPoolValue.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVDeadRegisterDefinitions.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMakeCompressible.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVExpandAtomicPseudoInsts.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVExpandPseudoInsts.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVFrameLowering.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVGatherScatterLowering.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVIndirectBranchTracking.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVInsertVSETVLI.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVInsertReadWriteCSR.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVInsertWriteVXRM.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVInstrInfo.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVISelDAGToDAG.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVISelLowering.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVLandingPadSetup.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMachineFunctionInfo.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMergeBaseOffset.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVOptWInstrs.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVPostRAExpandPseudoInsts.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVRedundantCopyElimination.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMoveMerger.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVPushPopOptimizer.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVRegisterInfo.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVSelectionDAGInfo.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVSubtarget.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVTargetMachine.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVTargetObjectFile.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVTargetTransformInfo.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVVectorMaskDAGMutation.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVVectorPeephole.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVVLOptimizer.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVZacasABIFix.cpp

  LINK_COMPONENTS
  BuddyAnalysis
  BuddyAsmPrinter
  BuddyCodeGen
  CodeGenTypes
  BuddyCore
  BuddyIPO
  BuddyGlobalISel
  MC
  BuddyRISCVDesc
  BuddyRISCVInfo
  BuddySelectionDAG
  Support
  BuddyTarget
  BuddyTransformUtils

  TargetParser

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
  BuddyRISCVDesc
  BuddyRISCVInfo
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
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVAsmPrinter.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVCallingConv.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVCodeGenPrepare.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVConstantPoolValue.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVDeadRegisterDefinitions.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVExpandAtomicPseudoInsts.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVExpandPseudoInsts.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVFrameLowering.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVGatherScatterLowering.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVISelDAGToDAG.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVLandingPadSetup.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMergeBaseOffset.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVMoveMerger.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVOptWInstrs.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVPostRAExpandPseudoInsts.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVPushPopOptimizer.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVRedundantCopyElimination.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVRegisterInfo.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVSelectionDAGInfo.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVSubtarget.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVTargetMachine.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVTargetObjectFile.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVTargetTransformInfo.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVVectorMaskDAGMutation.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVVectorPeephole.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVZacasABIFix.cpp


  # Add *.h files to track the copies above.
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVCallingConv.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVInstrInfo.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVRegisterInfo.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVSelectionDAGInfo.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVSubtarget.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVTargetMachine.h
  ${CMAKE_CURRENT_BINARY_DIR}/RISCVTargetObjectFile.h


  LINK_COMPONENTS
  MC
  BuddyRISCVInfo
  Support
  TargetParser

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
