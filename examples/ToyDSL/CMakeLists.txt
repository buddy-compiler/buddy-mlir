# Necessary configuration for Antlr.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

antlr_target(Toy 
  Toy.g4
  VISITOR)

add_subdirectory(include)

# Include toy definition directories. 
include_directories(${ANTLR_Toy_OUTPUT_DIR})
message("ANTLR_Toy_OUTPUT_DIR = ${ANTLR_Toy_OUTPUT_DIR}")
include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(include/toy)
include_directories(${CMAKE_CURRENT_BINARY_DIR})  
include_directories(${ANTLR4_INCLUDE_DIRS})
# use mlirtablen to generate toycombine code.

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCombineIncGen)

# Add DSL -> MLIR frontend.
add_llvm_executable(buddy-toy-dsl 
  main.cpp 
  mlir/Dialect.cpp
  mlir/LowerToAffineLoops.cpp
  mlir/LowerToLLVM.cpp
  mlir/ToyCombine.cpp  
  ${ANTLR_Toy_CXX_OUTPUTS}
  DEPENDS
  ToyCombineIncGen
  ToyDSLOpsIncGen)

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
  


