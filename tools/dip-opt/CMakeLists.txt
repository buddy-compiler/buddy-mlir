get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(dip-opt dip-opt.cpp)

target_link_libraries(dip-opt
  PRIVATE 
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  BuddyDIP
  LowerDIPPass
  )
