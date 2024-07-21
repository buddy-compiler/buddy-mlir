get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_tool(buddy-opt
  buddy-opt.cpp
)

target_link_libraries(buddy-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIROptLib
  CBConvVectorization
  BuddyBud
  BuddySche
  LowerBudPass
  BuddyDIP
  LowerDIPPass
  BuddyDAP
  LowerDAPPass
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
