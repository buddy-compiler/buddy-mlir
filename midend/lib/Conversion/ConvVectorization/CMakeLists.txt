add_mlir_library(CBConvVectorization
  CBConvVectorization.cpp
  GEMMPointwiseConv2DNhwcHwcf.cpp
  PoolingVectorization.cpp
  PoolingNhwcMaxVectorization.cpp
  Conv2dNhwcFhwcVectorization.cpp
  
  LINK_LIBS PUBLIC
  BuddyUtils
  )
