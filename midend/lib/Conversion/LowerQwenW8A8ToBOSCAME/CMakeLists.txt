add_mlir_library(LowerQwenW8A8ToBOSCAMEPass
  LowerQwenW8A8ToBOSCAME.cpp

  LINK_LIBS PUBLIC
  BuddyBOSCAME
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTosaDialect
  MLIRTransforms
  MLIRVectorDialect
)
