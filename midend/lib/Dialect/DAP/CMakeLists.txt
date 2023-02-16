add_mlir_dialect_library(BuddyDAP
  DAPDialect.cpp
  DAPOps.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  )
