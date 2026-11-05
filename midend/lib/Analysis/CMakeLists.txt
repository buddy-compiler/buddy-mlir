add_mlir_library(BuddyKernelContract
  KernelContract.cpp
  KernelContractCore.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
)

add_mlir_library(BuddyBOSCAMEAnalysis
  BOSCAMELaunchFacts.cpp
  LinalgComputationFacts.cpp
  LinalgComputationFactsPrinter.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRSideEffectInterfaces
)
