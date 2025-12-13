add_mlir_library(LowerIMEPass
  LowerIMEPass.cpp

  LINK_LIBS PUBLIC
  BuddyIMETransforms
)
