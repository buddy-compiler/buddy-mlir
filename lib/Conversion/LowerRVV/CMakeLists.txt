add_mlir_library(LowerRVVPass
  LowerRVVPass.cpp

  LINK_LIBS PUBLIC
  BuddyRVVTransforms
  )
