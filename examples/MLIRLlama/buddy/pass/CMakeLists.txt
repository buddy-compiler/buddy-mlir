add_mlir_library(MatMulOpt
	MatMulOptimize.cc
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_tool(llama-opt
  llama-opt.cc
)

target_link_libraries(llama-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  MatMulOpt
  )
