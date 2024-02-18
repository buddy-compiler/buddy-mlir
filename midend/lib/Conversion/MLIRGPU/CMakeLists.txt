add_mlir_library(MLIRGPUPasses
  GPUHostRegister.cpp
  ConvertMemcpyToGPU.cpp
  AsyncRegion.cpp
  MergeGPUAlloc.cpp

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
