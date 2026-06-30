add_subdirectory(Dialect)

set(LLVM_OPTIONAL_SOURCES
  ConvertBuddyToLLVMIR.cpp
  )

set(LinkedLibs
  BuddyRVVToLLVMIRTranslation
  BuddyGemminiToLLVMIRTranslation
  MLIRIMEToLLVMIRTranslation
  MLIRAMEToLLVMIRTranslation
  MLIRXTAMEToLLVMIRTranslation
  MLIRBOSCAMEToLLVMIRTranslation
  )

if(BUDDY_ENABLE_EXTERNAL_DIALECTS)
  list(APPEND LinkedLibs
    BuddyBuckyballToLLVMIRTranslation
    )
endif()

add_mlir_translation_library(BuddyToLLVMIRTranslationRegistration
  ConvertBuddyToLLVMIR.cpp

  LINK_LIBS PUBLIC
  ${LinkedLibs}
  )
