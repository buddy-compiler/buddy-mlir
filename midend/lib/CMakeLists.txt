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
  LowerSche
  LowerVectorExpPass
  MatMulOptimization
  BatchMatMulOptimization
  MatMulParallelVectorization
  SchedulingOnDevices
  TransposeOptimization
)


add_mlir_library(BuddyMLIRInitAll
  InitAll.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ${LinkedLibs}
)


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
