add_mlir_dialect_library(BuddyRVProf
  RVProfDialect.cpp
  RVProfOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/midend/include/Dialect/RVProf

  DEPENDS
  MLIRRVProfOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
