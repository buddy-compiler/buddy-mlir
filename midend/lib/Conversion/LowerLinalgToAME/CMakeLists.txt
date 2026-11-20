add_mlir_library(LowerLinalgToAMEPass
  LowerLinalgToAME.cpp

  LINK_LIBS PUBLIC
  BuddyAME
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
)
