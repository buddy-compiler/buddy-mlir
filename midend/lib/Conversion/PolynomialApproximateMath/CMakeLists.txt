add_mlir_library(PolynomialApproximateMathPass
  PolynomialApproximateMathPass.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRMathDialect
  MLIRMathTransforms
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
