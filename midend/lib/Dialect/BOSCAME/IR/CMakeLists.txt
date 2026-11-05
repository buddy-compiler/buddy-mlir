add_mlir_dialect_library(BuddyBOSCAME
  BOSCAMEDialect.cpp
  TileOps.cpp
  Endpoint.cpp

  # TableGen emits the SPM attribute definitions into a separate generated
  # target, so this library must depend on it explicitly.
  DEPENDS
  BuddyBOSCAMEAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
)
