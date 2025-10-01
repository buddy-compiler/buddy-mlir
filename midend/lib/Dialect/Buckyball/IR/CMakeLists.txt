add_mlir_dialect_library(BuddyBuckyball
  BuckyballDialect.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
)
