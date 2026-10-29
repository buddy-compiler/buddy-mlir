add_mlir_library(TosaElementwiseFusion
  SiLUFusion.cpp
  VectorExpApproximation.cpp

  LINK_LIBS PUBLIC
  BuddyUtils
  MLIRTosaDialect
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRMathTransforms
  MLIRLinalgTransforms
  MLIRVectorDialect
)
