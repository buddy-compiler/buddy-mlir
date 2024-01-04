# ExecutionEngine

set(MLIR_ExecutionEngine_DIR ${LLVM_PROJECT_SOURCE_DIR}/mlir/lib/ExecutionEngine)

add_mlir_library(MLIRCRunnerUtils
  ${MLIR_ExecutionEngine_DIR}/CRunnerUtils.cpp
  ${MLIR_ExecutionEngine_DIR}/SparseTensorRuntime.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  mlir_float16_utils
  MLIRSparseTensorEnums
  MLIRSparseTensorRuntime
  )
set_property(TARGET MLIRCRunnerUtils PROPERTY CXX_STANDARD 17)
target_compile_definitions(MLIRCRunnerUtils PRIVATE MLIRCRunnerUtils_EXPORTS)

add_mlir_library(MLIRRunnerUtils
  ${MLIR_ExecutionEngine_DIR}/RunnerUtils.cpp

  EXCLUDE_FROM_LIBMLIR
  )
target_compile_definitions(MLIRRunnerUtils PRIVATE MLIRRunnerUtils_EXPORTS)
