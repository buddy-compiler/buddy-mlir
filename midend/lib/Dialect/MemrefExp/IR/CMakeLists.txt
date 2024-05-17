add_mlir_dialect_library(BuddyMemrefExp
  MemrefExpDialect.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
)
