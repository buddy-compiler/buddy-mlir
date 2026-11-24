add_mlir_library(TosaElementwiseFusion
  SiLUFusion.cpp
  AttentionSoftmaxFusion.cpp
  VectorExpApproximation.cpp

  LINK_LIBS PUBLIC
  BuddyUtils
  MLIRTosaDialect
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRMathTransforms
  MLIRLinalgTransforms
  MLIRVectorDialect
)
