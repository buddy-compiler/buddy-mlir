add_mlir_dialect_library(BuddyDIP
  DIPDialect.cpp
  DIPOps.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  )
