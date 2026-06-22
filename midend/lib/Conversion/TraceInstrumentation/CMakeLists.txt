add_mlir_library(TraceInstrumentation
  InsertTensorTrace.cpp

  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRTensorDialect
)
