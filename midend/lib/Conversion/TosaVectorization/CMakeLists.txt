add_mlir_library(TosaVectorization
  ReduceSumVectorization3D.cpp
  
  LINK_LIBS PUBLIC
  BuddyUtils
)
