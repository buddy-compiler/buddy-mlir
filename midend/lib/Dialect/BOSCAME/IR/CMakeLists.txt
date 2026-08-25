add_mlir_dialect_library(BuddyBOSCAME
  BOSCAMEDialect.cpp

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRIR
  MLIRDestinationStyleOpInterface
)
