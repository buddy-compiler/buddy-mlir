add_mlir_library(LowerLinalgToMagiaPass
  LowerLinalgToMagia.cpp

  DEPENDS
  MLIRMagiaIncGen

  LINK_LIBS PUBLIC
  BuddyMagia
)
