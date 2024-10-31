add_mlir_dialect_library(BuddyLinalgPasses
  LinalgPromotion.cpp

  DEPENDS
  BuddyLinalgPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformDialect
)
