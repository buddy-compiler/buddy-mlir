add_subdirectory(profiler)
if(BUDDY_MLIR_ENABLE_PYTHON_PACKAGES)
  add_subdirectory(Python)
endif()

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_tool(buddy-profiler
  buddy-profiler.cpp
)

target_link_libraries(buddy-profiler
  PRIVATE
  BuddyRuntime
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

