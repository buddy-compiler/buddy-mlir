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
add_public_tablegen_target(FEGENToyCombineIncGen)

# Add DSL -> MLIR frontend.
add_llvm_executable(buddy-frontendgen-dsl 
  main.cpp
  ${LLVM_MLIR_SOURCE_DIR}/examples/toy/Ch7/mlir/LowerToAffineLoops.cpp
  ${LLVM_MLIR_SOURCE_DIR}/examples/toy/Ch7/mlir/Dialect.cpp
  ${LLVM_MLIR_SOURCE_DIR}/examples/toy/Ch7/mlir/LowerToLLVM.cpp
  ${LLVM_MLIR_SOURCE_DIR}/examples/toy/Ch7/mlir/ShapeInferencePass.cpp
  ${LLVM_MLIR_SOURCE_DIR}/examples/toy/Ch7/mlir/ToyCombine.cpp  
  ${ANTLR_Toy_CXX_OUTPUTS}
  DEPENDS
  FEGENToyCombineIncGen
  FEGENToyDSLOpsIncGen
  FEGENToyShapeInferenceInterfaceIncGen)

llvm_update_compile_flags(buddy-frontendgen-dsl)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# Add link libraries.
target_link_libraries(buddy-frontendgen-dsl 
  PRIVATE
  Threads::Threads
  MLIRAnalysis
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRExecutionEngine
  MLIRIR
  MLIRAffineDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTargetLLVMIRExport
  MLIRTransforms
  MLIRSupport
  MLIRLLVMCommonConversion
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  MLIRAffineToStandard
  antlr4_static
  ${dialect_libs}
  ${conversion_libs})
