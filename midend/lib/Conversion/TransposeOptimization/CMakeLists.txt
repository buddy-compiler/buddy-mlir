add_mlir_library(TransposeOptimization
  BuiltinTransposeVectorization.cpp
  GenericOpTransposeVectorization.cpp
  
  LINK_LIBS PUBLIC
  BuddyUtils
)
