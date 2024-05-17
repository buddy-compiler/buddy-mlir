add_mlir_library(FinalizeMemrefExpToLLVMPass
  FinalizeMemrefToLLVMPass.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRTransforms
  )
