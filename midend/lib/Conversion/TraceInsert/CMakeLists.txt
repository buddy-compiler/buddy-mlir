add_mlir_library(TraceInsert
  InsertTensorTrace.cpp

  LINK_LIBS PUBLIC
    BuddyTrace
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRPass
)
