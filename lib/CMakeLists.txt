add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Target)
add_subdirectory(Interface)

add_mlir_library(static_mlir_async_runtime
  STATIC
  ${LLVM_MLIR_SOURCE_DIR}/lib/ExecutionEngine/AsyncRuntime.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  ${LLVM_PTHREAD_LIB}
)

target_compile_definitions(static_mlir_async_runtime
  PRIVATE
  MLIR_ASYNCRUNTIME_DEFINE_FUNCTIONS
  )
