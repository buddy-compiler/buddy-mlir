# include_directories(../include)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)


add_library(BuddyRuntime STATIC
  Interface.cpp
  TimeEvent.cpp
  TimeManager.cpp
  Profiler.cpp
  )

target_link_libraries(BuddyRuntime
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIROptLib
  CBConvVectorization
  BuddyBud
  BuddyTrace
  LowerTracePass
  BuddySche
  LowerBudPass
  BuddyDIP
  LowerDIPPass
  BuddyDAP
  LowerDAPPass
  ExtendDAPPass
  DAPVectorization
  BuddyRVV
  LowerRVVPass
  MatMulOptimization
  BatchMatMulOptimization
  MatMulParallelVectorization
  TransposeOptimization
  ConvOptimization
  VectorExp
  LowerVectorExpPass
  BuddyGemmini
  LowerGemminiPass
  LowerLinalgToGemminiPass
  SchedulingOnDevices
  LowerSche
  FuncBufferizeDynamicOffset
)

