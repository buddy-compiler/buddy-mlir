add_mlir_dialect_library(BuddyBud
  BudDialect.cpp
  BudOps.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  )
