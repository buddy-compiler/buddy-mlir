get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(test_libs
  MLIRGPUTestPasses
)

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
  ${test_libs}
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
  BuddySche
  VectorExp
  BuddyMLIRInitAll
  BuddyToLLVMIRTranslationRegistration
)
