add_mlir_dialect_library(BuddySche
  ScheDialect.cpp
  ScheOps.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  )
