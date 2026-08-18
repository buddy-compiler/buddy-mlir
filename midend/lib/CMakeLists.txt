add_subdirectory(CAPI)
add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(ExecutionEngine)
add_subdirectory(Target)
add_subdirectory(Utils)


get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LinkedLibs
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  ${extension_libs}

  BuddyBOSCAME
  BuddyBOSCAMETransforms
  BuddyXTAME
  BuddyXTAMETransforms
  BuddyAME
  BuddyAMETransforms
  BuddyRHAL
  BuddyTile
  ConvOptimization
  CBConvVectorization
  LowerBudPass
  LowerDAPPass
  LowerDIPPass
  LowerLinalgToTilePass
  LowerGemminiPass
  LowerLinalgToBOSCAMEPass
  LowerLinalgToGemminiPass
  LowerLinalgToIMEPass
  LowerIMEPass
  LowerRVVPass
  LowerVectorExpPass
  MatMulOptimization
  BatchMatMulOptimization
  MatMulParallelVectorization
  TransposeOptimization
  TosaElementwiseFusion
  AssumeTightMemRefLayoutPass
  StaticizeMemRefLayoutPass
  EliminateMemRefCopyPass
  EliminateLargeZeroConstants
  TraceInsert
  GraphRedundancyElimination
)

if(BUDDY_ENABLE_EXTERNAL_DIALECTS)
  list(APPEND LinkedLibs
    LowerTileToBuckyballPass
    LowerBuckyballPass
  )
endif()

add_mlir_library(BuddyMLIRInitAll
  InitAll.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ${LinkedLibs}
)
