add_mlir_library(TransposeOptimization
  BuiltinTransposeVectorization.cpp
  TransposeFusionVectorization.cpp
  LINK_LIBS PUBLIC
  BuddyUtils
)
