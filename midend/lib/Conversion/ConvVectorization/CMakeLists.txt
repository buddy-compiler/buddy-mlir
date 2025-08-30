add_mlir_library(CBConvVectorization
  CBConvVectorization.cpp
  GEMMPointwiseConv2DNhwcHwcf.cpp
  PoolingVectorization.cpp
  PoolingNhwcMaxVectorization.cpp
  
  LINK_LIBS PUBLIC
  BuddyUtils
  )
