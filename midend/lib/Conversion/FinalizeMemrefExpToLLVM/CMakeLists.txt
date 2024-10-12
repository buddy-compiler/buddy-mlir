add_mlir_library(FinalizeMemrefExpToLLVMPass
  FinalizeMemrefToLLVMPass.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  )
