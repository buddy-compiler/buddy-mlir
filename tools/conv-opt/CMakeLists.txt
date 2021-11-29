get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_tool(conv-opt
  conv-opt.cpp
)

target_link_libraries(conv-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  CBConvVectorization
  )
