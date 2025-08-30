# Configuration for ANTLR.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

antlr_target(Toy 
  Toy.g4
  VISITOR)

# Include toy definition directories. 
include_directories(${ANTLR_Toy_OUTPUT_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/toy) 
include_directories(${CMAKE_CURRENT_BINARY_DIR}) 
include_directories(${ANTLR4_INCLUDE_DIRS})
include_directories(${LLVM_MLIR_SOURCE_DIR}/examples/toy/Ch7/include)
include_directories(include)

add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  OrcJIT
  )

set(LLVM_TARGET_DEFINITIONS ${LLVM_MLIR_SOURCE_DIR}/examples/toy/Ch7/mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCombineIncGen)

# Add DSL -> MLIR frontend.
add_llvm_executable(buddy-toy-dsl 
  main.cpp
  ${LLVM_MLIR_SOURCE_DIR}/examples/toy/Ch7/mlir/LowerToAffineLoops.cpp
  ${LLVM_MLIR_SOURCE_DIR}/examples/toy/Ch7/mlir/Dialect.cpp
  ${LLVM_MLIR_SOURCE_DIR}/examples/toy/Ch7/mlir/LowerToLLVM.cpp
  ${LLVM_MLIR_SOURCE_DIR}/examples/toy/Ch7/mlir/ShapeInferencePass.cpp
  ${LLVM_MLIR_SOURCE_DIR}/examples/toy/Ch7/mlir/ToyCombine.cpp  
  ${ANTLR_Toy_CXX_OUTPUTS}
  DEPENDS
  ToyCombineIncGen
  ToyDSLOpsIncGen
  ToyShapeInferenceInterfaceIncGen)

llvm_update_compile_flags(buddy-toy-dsl)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

# Add link libraries.
target_link_libraries(buddy-toy-dsl 
  PRIVATE
  Threads::Threads
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIRAnalysis
  MLIRBuiltinToLLVMIRTranslation
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRExecutionEngine
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTargetLLVMIRExport
  MLIRTransforms
  antlr4_static
  ${dialect_libs}
  ${conversion_libs})
