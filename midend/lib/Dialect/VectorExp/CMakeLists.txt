add_mlir_dialect_library(VectorExp
  VectorExpDialect.cpp
  VectorExpOps.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  )
