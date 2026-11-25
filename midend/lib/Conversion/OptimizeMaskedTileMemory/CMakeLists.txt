add_mlir_library(OptimizeMaskedTileMemoryPasses
  ForwardMaskedLoads.cpp
  OptimizeMaskedTileMemory.cpp
  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRMemRefUtils
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRVectorDialect
  MLIRPass
  MLIRTransformUtils
)
