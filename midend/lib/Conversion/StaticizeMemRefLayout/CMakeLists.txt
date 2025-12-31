add_mlir_library(StaticizeMemRefLayoutPass
  StaticizeMemRefLayout.cpp

  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
)
