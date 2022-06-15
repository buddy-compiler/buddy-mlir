# Necessary configuration for Antlr.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)


antlr_target(toy toy.g4
            VISITOR)
include_directories(${ANTLR_toy_OUTPUT_DIR})

# Include toy definition directories. 
include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(include/toy)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${ANTLR4_INCLUDE_DIRS})

# use mlirtablen to generate toycombine code.
set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCombineIncGen)

add_subdirectory(include)

# Add DSL -> MLIR frontend.
add_llvm_executable(buddy-toy-dsl 
										main.cpp 
										mlir/Dialect.cpp
										mlir/LowerToAffineLoops.cpp
										mlir/LowerToLLVM.cpp
										mlir/ToyCombine.cpp  
                    ${ANTLR_toy_CXX_OUTPUTS}
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
    ${conversion_libs}
)


