get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_tool(buddy-opt
  buddy-opt.cpp
)

target_link_libraries(buddy-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  CBConvVectorization
  BuddyBud
  LowerBudPass
  BuddyDIP
  LowerDIPPass
  BuddyDAP
  LowerDAPPass
  BuddyRVV
  LowerRVVPass
  MatMulOptimization
  ConvOptimization
  VectorExp
  LowerVectorExpPass
  )
