add_mlir_library(LowerRVVPass
  LowerRVVPass.cpp
  VectorToRVV.cpp

  LINK_LIBS PUBLIC
  BuddyRVVTransforms
  MLIRVectorDialect
  )
