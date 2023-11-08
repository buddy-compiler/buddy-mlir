add_mlir_public_c_api_library(BuddyMLIRCAPI
  Dialects.cpp
  Registration.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/midend/include/buddy-mlir-c

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  BuddyGemmini
  BuddyGemminiTransforms
  BuddyMLIRInitAll
)