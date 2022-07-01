# Necessary configuration for Antlr.
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
include_directories(../../llvm/mlir/examples/toy/Ch6/include)
include_directories(include)

add_subdirectory(include)

# Add DSL -> MLIR frontend.
add_llvm_executable(buddy-toy-dsl 
  main.cpp
  ../../llvm/mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp
  ../../llvm/mlir/examples/toy/Ch5/mlir/Dialect.cpp
  ../../llvm/mlir/examples/toy/Ch6/mlir/LowerToLLVM.cpp
  ../../llvm/mlir/examples/toy/Ch5/mlir/ToyCombine.cpp  
  ${ANTLR_Toy_CXX_OUTPUTS}
  DEPENDS
  ToyCombineIncGen
  ToyDSLOpsIncGen
  ToyShapeInferenceInterfaceIncGen)

llvm_update_compile_flags(buddy-toy-dsl)

# add linklibrary.
target_link_libraries(buddy-toy-dsl 
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
