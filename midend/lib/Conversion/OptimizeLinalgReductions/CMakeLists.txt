add_mlir_library(OptimizeLinalgReductionPasses
  OptimizeLinalgReductions.cpp
  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRPass
  MLIRTensorDialect
  MLIRTransformUtils
)
