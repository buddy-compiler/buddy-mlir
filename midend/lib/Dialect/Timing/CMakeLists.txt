add_mlir_dialect_library(BuddyTiming
  TimingDialect.cpp
  TimingOps.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  )
