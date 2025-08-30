add_mlir_library(BuddyUtils
  Utils.cpp
  DIPUtils.cpp
  DAPUtils.cpp
  GPUUtils.cpp
  AffineTransformUtils.cpp
  )

add_mlir_library(BuddyDIPUtils
  DIPUtils.cpp

  LINK_LIBS PUBLIC
  BuddyUtils
  )

add_mlir_library(BuddyDAPUtils
  DAPUtils.cpp
  
  LINK_LIBS PUBLIC
  BuddyUtils
  )

add_mlir_library(BuddyGPUUtils
  GPUUtils.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  LLVMTargetParser
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRAnalysis
  MLIRArithDialect
  MLIRArithUtils
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTensorDialect
  MLIRTilingInterface
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRViewLikeInterface
  MLIRGPUPasses
)
