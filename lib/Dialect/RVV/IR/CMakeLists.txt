add_mlir_dialect_library(BuddyRVV
  RVVDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_BINARY_DIR}/include/Dialect/RISCVV

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  )
