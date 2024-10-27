set(LLVM_Object_DIR ${LLVM_MAIN_SRC_DIR}/lib/Object)

add_llvm_component_library(LLVMBuddyObject
  ${LLVM_Object_DIR}/Archive.cpp
  ${LLVM_Object_DIR}/ArchiveWriter.cpp
  ${LLVM_Object_DIR}/Binary.cpp
  ${LLVM_Object_DIR}/BuildID.cpp
  ${LLVM_Object_DIR}/COFFImportFile.cpp
  ${LLVM_Object_DIR}/COFFModuleDefinition.cpp
  ${LLVM_Object_DIR}/COFFObjectFile.cpp
  ${LLVM_Object_DIR}/Decompressor.cpp
  ${LLVM_Object_DIR}/DXContainer.cpp
  ${LLVM_Object_DIR}/ELF.cpp
  ${LLVM_Object_DIR}/ELFObjectFile.cpp
  ${LLVM_Object_DIR}/Error.cpp
  ${LLVM_Object_DIR}/GOFFObjectFile.cpp
  ${LLVM_Object_DIR}/FaultMapParser.cpp
  ${LLVM_Object_DIR}/IRObjectFile.cpp
  ${LLVM_Object_DIR}/IRSymtab.cpp
  ${LLVM_Object_DIR}/MachOObjectFile.cpp
  ${LLVM_Object_DIR}/MachOUniversal.cpp
  ${LLVM_Object_DIR}/Minidump.cpp
  ${LLVM_Object_DIR}/ModuleSymbolTable.cpp
  ${LLVM_Object_DIR}/Object.cpp
  ${LLVM_Object_DIR}/ObjectFile.cpp
  ${LLVM_Object_DIR}/OffloadBinary.cpp
  ${LLVM_Object_DIR}/RecordStreamer.cpp
  ${LLVM_Object_DIR}/RelocationResolver.cpp
  ${LLVM_Object_DIR}/SymbolicFile.cpp
  ${LLVM_Object_DIR}/SymbolSize.cpp
  ${LLVM_Object_DIR}/TapiFile.cpp
  ${LLVM_Object_DIR}/TapiUniversal.cpp
  ${LLVM_Object_DIR}/MachOUniversalWriter.cpp
  ${LLVM_Object_DIR}/WasmObjectFile.cpp
  ${LLVM_Object_DIR}/WindowsMachineFlag.cpp
  ${LLVM_Object_DIR}/WindowsResource.cpp
  ${LLVM_Object_DIR}/XCOFFObjectFile.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Object

  DEPENDS
  buddy_intrinsics_gen

  LINK_COMPONENTS
  BuddyBitReader
  BuddyCore
  MC
  BuddyIRReader
  BinaryFormat
  MCParser
  Support
  TargetParser
  TextAPI
  )
