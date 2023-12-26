add_mlir_library(DAPVectorization
  DAPVectorization.cpp

  LINK_LIBS PUBLIC
  BuddyDAPUtils
)
