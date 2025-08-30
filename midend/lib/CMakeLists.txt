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

  ConvOptimization
  CBConvVectorization
  LowerBudPass
  LowerDAPPass
  LowerDIPPass
  LowerGemminiPass
  LowerLinalgToGemminiPass
  LowerRVVPass
  LowerVectorExpPass
  MatMulOptimization
  BatchMatMulOptimization
  MatMulParallelVectorization
  TransposeOptimization
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

# Build static library for MLIR C runner utils runtime.
add_mlir_library(StaticMLIRCRunnerUtils
  ${MLIR_MAIN_SRC_DIR}/lib/ExecutionEngine/CRunnerUtils.cpp
  ${MLIR_MAIN_SRC_DIR}/lib/ExecutionEngine/SparseTensorRuntime.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  mlir_float16_utils
  MLIRSparseTensorEnums
  MLIRSparseTensorRuntime
  )
set_property(TARGET StaticMLIRCRunnerUtils PROPERTY CXX_STANDARD 17)
target_compile_definitions(StaticMLIRCRunnerUtils PRIVATE StaticMLIRCRunnerUtils_EXPORTS)

# Build static library for MLIR runner utils runtime.
add_mlir_library(StaticMLIRRunnerUtils
  ${MLIR_MAIN_SRC_DIR}/lib/ExecutionEngine/RunnerUtils.cpp

  EXCLUDE_FROM_LIBMLIR
  )
target_compile_definitions(StaticMLIRRunnerUtils PRIVATE StaticMLIRRunnerUtils_EXPORTS)
