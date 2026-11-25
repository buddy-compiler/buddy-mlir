add_mlir_library(LowerMagiaPass
  LowerMagiaPass.cpp

  DEPENDS
  MLIRMagiaIncGen

  LINK_LIBS PUBLIC
  BuddyMagia
)
