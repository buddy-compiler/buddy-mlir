add_subdirectory(Dialect)

set(LLVM_OPTIONAL_SOURCES
  ConvertBuddyToLLVMIR.cpp
  )

add_mlir_translation_library(BuddyToLLVMIRTranslationRegistration
  ConvertBuddyToLLVMIR.cpp

  LINK_LIBS PUBLIC
  BuddyRVVToLLVMIRTranslation
  BuddyGemminiToLLVMIRTranslation
  )
