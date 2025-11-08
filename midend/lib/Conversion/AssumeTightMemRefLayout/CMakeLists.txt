add_mlir_library(AssumeTightMemRefLayoutPass
  AssumeTightMemRefLayout.cpp

  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
)
