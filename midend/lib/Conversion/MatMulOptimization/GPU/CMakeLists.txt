add_mlir_library(BuddyIRGPUMatmulOpt
  DEPENDS
  MhloDialect
  MLIRBufferTransforms

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransforms
  MLIRLinalgExtTransformOps
  BuddyUtils
)