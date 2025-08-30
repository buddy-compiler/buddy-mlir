set(LLVM_ProfileData_DIR ${LLVM_MAIN_SRC_DIR}/lib/ProfileData)

add_llvm_component_library(LLVMBuddyProfileData
  ${LLVM_ProfileData_DIR}/GCOV.cpp
  ${LLVM_ProfileData_DIR}/InstrProf.cpp
  ${LLVM_ProfileData_DIR}/InstrProfCorrelator.cpp
  ${LLVM_ProfileData_DIR}/InstrProfReader.cpp
  ${LLVM_ProfileData_DIR}/InstrProfWriter.cpp
  ${LLVM_ProfileData_DIR}/ItaniumManglingCanonicalizer.cpp
  ${LLVM_ProfileData_DIR}/MemProf.cpp
  ${LLVM_ProfileData_DIR}/MemProfReader.cpp
  ${LLVM_ProfileData_DIR}/PGOCtxProfReader.cpp
  ${LLVM_ProfileData_DIR}/PGOCtxProfWriter.cpp
  ${LLVM_ProfileData_DIR}/ProfileSummaryBuilder.cpp
  ${LLVM_ProfileData_DIR}/SampleProf.cpp
  ${LLVM_ProfileData_DIR}/SampleProfReader.cpp
  ${LLVM_ProfileData_DIR}/SampleProfWriter.cpp
  ${LLVM_ProfileData_DIR}/SymbolRemappingReader.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ProfileData

  DEPENDS
  buddy_intrinsics_gen

  LINK_COMPONENTS
  BuddyCore
  BuddyObject
  Object
  Support
  Demangle
  Symbolize
  DebugInfoDWARF
  TargetParser
  )
