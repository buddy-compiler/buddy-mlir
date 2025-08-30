add_mlir_library(FuncBufferizeDynamicOffset
FuncBufferizePass.cpp

LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRTransforms
)
