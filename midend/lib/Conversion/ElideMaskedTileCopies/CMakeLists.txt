add_mlir_library(ElideMaskedTileCopiesPass
  ElideMaskedTileCopies.cpp
  LINK_LIBS PUBLIC
  MLIRArithDialect MLIRFuncDialect MLIRLinalgDialect MLIRMemRefDialect
  MLIRSCFDialect MLIRVectorDialect MLIRPass MLIRTransformUtils
)
