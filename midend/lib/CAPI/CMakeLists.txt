get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

link_directories(${LLVM_BINARY_DIR}/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs/)

add_mlir_public_c_api_library(BuddyMLIRCAPI
  Dialects.cpp
  RegisterEverything.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/midend/include/buddy-mlir-c

  LINK_LIBS PUBLIC
  ${dialect_libs}
  ${translation_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRBuiltinToLLVMIRTranslation
  MLIRCAPIIR
  MLIRLLVMToLLVMIRTranslation
  MLIRCAPITransforms
  BuddyBud
  BuddyDAP
  BuddyDIP
  BuddyGemmini
  BuddyGemminiTransforms
  BuddyRVV
  BuddyRVVTransforms
  VectorExp
  BuddyMLIRInitAll
  BuddyToLLVMIRTranslationRegistration
)

if(BUDDY_ENABLE_CUSTOM_BACKEND)
  target_compile_definitions(BuddyMLIRCAPI PRIVATE BUDDY_ENABLE_CUSTOM_BACKEND)
  if(TARGET obj.BuddyMLIRCAPI)
    target_compile_definitions(obj.BuddyMLIRCAPI PRIVATE BUDDY_ENABLE_CUSTOM_BACKEND)
  endif()
endif()
