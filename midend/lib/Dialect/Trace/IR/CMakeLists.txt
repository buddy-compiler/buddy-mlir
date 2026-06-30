add_mlir_dialect_library(BuddyTrace
  TraceDialect.cpp
  BufferizableOpInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRIR
  MLIRSideEffectInterfaces
  )
