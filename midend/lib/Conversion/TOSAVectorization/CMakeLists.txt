add_mlir_library(TOSAVectorization
  BuiltinReduceSumVectorization.cpp
  LINK_LIBS PUBLIC
  BuddyUtils
)