//====- conv-opt.cpp - Convolution Optimizations Main ========================//
//
// This file is the driver of convolution optimizers.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir {
namespace buddy {
void registerConvVectorizationPass();
} // namespace buddy
} // namespace mlir

int main(int argc, char **argv) {
  // Register all MLIR passes.
  mlir::registerAllPasses();
  // Register Vectorization of Convolution.
  mlir::buddy::registerConvVectorizationPass();

  mlir::DialectRegistry registry;
  // Register all MLIR dialects.
  registerAllDialects(registry);

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "Convolution optimizer driver", registry));
}