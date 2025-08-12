add_mlir_dialect_library(VIR
  VIRDialect.cpp
  VIROps.cpp
  VIRTypes.cpp
  VIRAttrs.cpp

  DEPENDS
  MLIRVIRAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  )
