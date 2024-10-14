add_mlir_library(BuddyGPUPipelines
  GemmCodegenTransform.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPDLDialect
  MLIRTransformDialect
  MLIRTransforms
  BuddyTransformPasses
  BuddyGemmCodegenUtils
)