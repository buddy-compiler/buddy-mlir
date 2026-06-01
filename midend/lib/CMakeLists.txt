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

  BuddyBOSCAME
  BuddyBOSCAMETransforms
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


# Build static library for async runtime.
add_mlir_library(static_mlir_async_runtime
  STATIC
  ${MLIR_MAIN_SRC_DIR}/lib/ExecutionEngine/AsyncRuntime.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  ${LLVM_PTHREAD_LIB}
)

target_compile_definitions(static_mlir_async_runtime
  PRIVATE
  MLIR_ASYNCRUNTIME_DEFINE_FUNCTIONS
  )

# Build static library for MLIR float16 utils runtime.
add_mlir_library(StaticMLIRFloat16Utils
  STATIC
  ${MLIR_MAIN_SRC_DIR}/lib/ExecutionEngine/Float16bits.cpp

  EXCLUDE_FROM_LIBMLIR
)
set_property(TARGET StaticMLIRFloat16Utils PROPERTY CXX_STANDARD 17)

# Build static library for MLIR sparse tensor runtime.
add_mlir_library(StaticMLIRSparseTensorRuntime
  STATIC
  ${MLIR_MAIN_SRC_DIR}/lib/ExecutionEngine/SparseTensor/File.cpp
  ${MLIR_MAIN_SRC_DIR}/lib/ExecutionEngine/SparseTensor/MapRef.cpp
  ${MLIR_MAIN_SRC_DIR}/lib/ExecutionEngine/SparseTensor/Storage.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRSparseTensorEnums
  StaticMLIRFloat16Utils
)
set_property(TARGET StaticMLIRSparseTensorRuntime PROPERTY CXX_STANDARD 17)

# Build static library for MLIR C runner utils runtime.
add_mlir_library(StaticMLIRCRunnerUtils
  STATIC
  ${MLIR_MAIN_SRC_DIR}/lib/ExecutionEngine/CRunnerUtils.cpp
  ${MLIR_MAIN_SRC_DIR}/lib/ExecutionEngine/SparseTensorRuntime.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  StaticMLIRFloat16Utils
  MLIRSparseTensorEnums
  StaticMLIRSparseTensorRuntime
  )
set_property(TARGET StaticMLIRCRunnerUtils PROPERTY CXX_STANDARD 17)
target_compile_definitions(StaticMLIRCRunnerUtils PRIVATE StaticMLIRCRunnerUtils_EXPORTS)

# Build static library for MLIR runner utils runtime.
add_mlir_library(StaticMLIRRunnerUtils
  STATIC
  ${MLIR_MAIN_SRC_DIR}/lib/ExecutionEngine/RunnerUtils.cpp

  EXCLUDE_FROM_LIBMLIR
  )
target_compile_definitions(StaticMLIRRunnerUtils PRIVATE StaticMLIRRunnerUtils_EXPORTS)
