add_subdirectory(Dialect)

set(LLVM_OPTIONAL_SOURCES
  ConvertBuddyToLLVMIR.cpp
  )

set(_buddy_to_llvmir_link_libs BuddyRVVToLLVMIRTranslation)
if(BUDDY_ENABLE_CUSTOM_BACKEND)
  list(APPEND _buddy_to_llvmir_link_libs
    BuddyGemminiToLLVMIRTranslation
    MLIRIMEToLLVMIRTranslation
    MLIRAMEToLLVMIRTranslation
    MLIRXTAMEToLLVMIRTranslation
    MLIRBOSCAMEToLLVMIRTranslation
  )
endif()

add_mlir_translation_library(BuddyToLLVMIRTranslationRegistration
  ConvertBuddyToLLVMIR.cpp

  LINK_LIBS PUBLIC
  ${_buddy_to_llvmir_link_libs}
  )

if(BUDDY_ENABLE_CUSTOM_BACKEND)
  target_compile_definitions(BuddyToLLVMIRTranslationRegistration PRIVATE BUDDY_ENABLE_CUSTOM_BACKEND)
  if(TARGET obj.BuddyToLLVMIRTranslationRegistration)
    target_compile_definitions(obj.BuddyToLLVMIRTranslationRegistration PRIVATE BUDDY_ENABLE_CUSTOM_BACKEND)
  endif()
endif()
