add_mlir_library(VectorizeDequantizePass
  VectorizeDequantize.cpp

  LINK_LIBS PUBLIC
    MLIRAnalysis
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
    MLIRVectorDialect
)
