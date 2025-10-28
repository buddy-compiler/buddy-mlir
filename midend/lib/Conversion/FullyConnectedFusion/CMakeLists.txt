add_mlir_library(FullyConnectedFusion
  FuseFC.cpp
  LINK_LIBS PUBLIC
  BuddyUtils
)
