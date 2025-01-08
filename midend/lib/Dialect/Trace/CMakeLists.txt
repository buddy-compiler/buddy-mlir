add_mlir_dialect_library(BuddyTrace
  TraceDialect.cpp
  TraceOps.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  )
