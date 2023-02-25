add_mlir_dialect_library(BuddyGemmini
  GemminiDialect.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
)
