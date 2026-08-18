add_mlir_library(MLIRGPUPasses
  ConvertMemcpyToGPU.cpp
  ConvertStridedCopyToParallel.cpp
  LegalizeShmemOutlining.cpp
  MatMulGPUTiling.cpp
  NestParallelLoops.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Bufferization

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRControlFlowInterfaces
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRSubsetOpInterface
  MLIRTransforms
  MLIRViewLikeInterface
  MLIRSupport
  BuddyUtils
  MLIRBufferizationTransforms
  MLIRGPUDialect
)
