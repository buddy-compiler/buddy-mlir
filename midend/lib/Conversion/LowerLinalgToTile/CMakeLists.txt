add_mlir_library(LowerLinalgToTilePass
  LowerLinalgToTile.cpp

  LINK_LIBS PUBLIC
  BuddyTile
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRPass
  MLIRTransformUtils
  )
