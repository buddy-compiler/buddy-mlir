add_mlir_library(LowerLinalgToBOSCAMEPass
  LowerLinalgToBOSCAME.cpp

  LINK_LIBS PUBLIC
  BuddyBOSCAME
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTransforms
)
