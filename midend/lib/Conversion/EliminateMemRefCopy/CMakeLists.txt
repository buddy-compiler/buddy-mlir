add_mlir_library(EliminateMemRefCopyPass
  EliminateMemRefCopy.cpp

  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRTransformUtils
)
