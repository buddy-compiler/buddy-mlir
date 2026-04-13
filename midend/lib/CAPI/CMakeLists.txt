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
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
)

# CMake 3.31+ may drop MLIRRegisterAll* from aggregate deps; force the archives
# at the front of MLIR_AGGREGATE_DEPS so RegisterAll*.o precedes dialect libs.
if(TARGET MLIRRegisterAllDialects)
  get_target_property(_buddy_capi_agg_deps BuddyMLIRCAPI MLIR_AGGREGATE_DEPS)
  set_property(TARGET BuddyMLIRCAPI PROPERTY MLIR_AGGREGATE_DEPS
    "$<TARGET_FILE:MLIRRegisterAllDialects>"
    "$<TARGET_FILE:MLIRRegisterAllExtensions>"
    "$<TARGET_FILE:MLIRRegisterAllPasses>"
    ${_buddy_capi_agg_deps}
  )
endif()
