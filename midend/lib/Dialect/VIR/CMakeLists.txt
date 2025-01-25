add_mlir_dialect_library(VIR
  VIRDialect.cpp
  VIROps.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  )
