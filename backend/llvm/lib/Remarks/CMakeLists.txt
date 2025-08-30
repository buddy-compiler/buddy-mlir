set(LLVM_Remarks_DIR ${LLVM_MAIN_SRC_DIR}/lib/Remarks)

add_llvm_component_library(LLVMBuddyRemarks
  ${LLVM_Remarks_DIR}/BitstreamRemarkParser.cpp
  ${LLVM_Remarks_DIR}/BitstreamRemarkSerializer.cpp
  ${LLVM_Remarks_DIR}/Remark.cpp
  ${LLVM_Remarks_DIR}/RemarkFormat.cpp
  ${LLVM_Remarks_DIR}/RemarkLinker.cpp
  ${LLVM_Remarks_DIR}/RemarkParser.cpp
  ${LLVM_Remarks_DIR}/RemarkSerializer.cpp
  ${LLVM_Remarks_DIR}/RemarkStreamer.cpp
  ${LLVM_Remarks_DIR}/RemarkStringTable.cpp
  ${LLVM_Remarks_DIR}/YAMLRemarkParser.cpp
  ${LLVM_Remarks_DIR}/YAMLRemarkSerializer.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Remarks

  LINK_COMPONENTS
  BitstreamReader
  Support

  DEPENDS
  buddy_intrinsics_gen
  )
