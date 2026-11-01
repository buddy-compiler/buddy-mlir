add_mlir_library(OutlineLoopNestsPass
  OutlineLoopNests.cpp

  LINK_LIBS PUBLIC
    MLIRFuncDialect
    MLIRIR
    MLIRPass
    MLIRSCFDialect
    MLIRSCFUtils
)
