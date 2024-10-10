add_mlir_dialect_library(BuddyGPUPasses
  GPUDistributeToWarp.cpp
  RemoveReduntantLoops.cpp

  DEPENDS
  BuddyGPUPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)
