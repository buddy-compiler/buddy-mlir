add_mlir_dialect_library(BuddyRHAL
  RHALDialect.cpp
  RHALOps.cpp

  DEPENDS
  MLIRRHALOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
