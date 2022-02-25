//====- buddy-opt.cpp - The driver of buddy-mlir --------------------------===//
//
// This file is the dialects and oprimization driver of buddy-mlir project.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Bud/BudDialect.h"
#include "Bud/BudOps.h"
#include "DIP/DIPDialect.h"
#include "DIP/DIPOps.h"
#include "RVV/RVVDialect.h"

namespace mlir {
namespace buddy {
void registerConvVectorizationPass();
void registerPointwiseConvToGemmPass();
void registerPoolingVectorizationPass();
void registerLowerBudPass();
void registerLowerDIPPass();
void registerLowerRVVPass();
} // namespace buddy
} // namespace mlir

int main(int argc, char **argv) {
  // Register all MLIR passes.
  mlir::registerAllPasses();
  mlir::buddy::registerPointwiseConvToGemmPass();
  // Register Vectorization of Convolution.
  mlir::buddy::registerConvVectorizationPass();
  // Register Vectorization of Pooling.
  mlir::buddy::registerPoolingVectorizationPass();
  mlir::buddy::registerLowerBudPass();
  mlir::buddy::registerLowerDIPPass();
  mlir::buddy::registerLowerRVVPass();

  mlir::DialectRegistry registry;
  // Register all MLIR core dialects.
  registerAllDialects(registry);
  // Register dialects in buddy-mlir project.
  // clang-format off
  registry.insert<buddy::bud::BudDialect,
                  buddy::dip::DIPDialect,
                  buddy::rvv::RVVDialect>();
  // clang-format on

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "buddy-mlir optimizer driver", registry));
}
