add_mlir_library(TosaVectorization
  ReduceSumVectorization.cpp
  
  LINK_LIBS PUBLIC
  BuddyUtils
)
