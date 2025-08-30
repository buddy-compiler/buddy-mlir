add_mlir_library(TransposeOptimization
  BuiltinTransposeVectorization.cpp
  LINK_LIBS PUBLIC
  BuddyUtils
)
