add_mlir_library(TosaElementwiseFusion
  SiLUFusion.cpp

  LINK_LIBS PUBLIC
  BuddyUtils
  MLIRTosaDialect
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRTransforms
)
