add_subdirectory(CAPI)
add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Target)
add_subdirectory(Utils)


get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LinkedLibs
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  ${extension_libs}

  BuddyXTAME
  BuddyXTAMETransforms
  BuddyAME
  BuddyAMETransforms
  BuddyRHAL
  ConvOptimization
  CBConvVectorization
  LowerBudPass
  LowerDAPPass
  LowerDIPPass
  LowerGemminiPass
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
  GraphRedundancyElimination
)


add_mlir_library(BuddyMLIRInitAll
  InitAll.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ${LinkedLibs}
)


if(NOT TARGET mlir_async_runtime OR
   NOT TARGET mlir_c_runner_utils OR
   NOT TARGET mlir_runner_utils)
  message(FATAL_ERROR
    "MLIR runtime targets are missing from the configured MLIR package. "
    "buddy-mlir now requires an install tree that exports "
    "mlir_async_runtime, mlir_c_runner_utils, and mlir_runner_utils.")
endif()

message(STATUS "Using installed MLIR runtime targets.")

add_library(static_mlir_async_runtime INTERFACE)
target_link_libraries(static_mlir_async_runtime INTERFACE mlir_async_runtime)

add_library(StaticMLIRCRunnerUtils INTERFACE)
target_link_libraries(StaticMLIRCRunnerUtils INTERFACE mlir_c_runner_utils)

add_library(StaticMLIRRunnerUtils INTERFACE)
target_link_libraries(StaticMLIRRunnerUtils INTERFACE mlir_runner_utils)
