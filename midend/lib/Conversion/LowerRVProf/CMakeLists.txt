add_mlir_library(LowerRVProfPass
  LowerRVProfPass.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  )
