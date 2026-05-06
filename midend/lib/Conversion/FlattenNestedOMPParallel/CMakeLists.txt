add_mlir_library(FlattenNestedOMPParallelPass
  FlattenNestedOMPParallel.cpp

  LINK_LIBS PUBLIC
    MLIRIR
    MLIROpenMPDialect
    MLIRPass
    MLIRSCFDialect
    MLIRTransformUtils
)
