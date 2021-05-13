//===- img-opt.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is the driver for image dialect.
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

#include "Img/ImgDialect.h"
#include "Img/ImgOps.h"

namespace mlir {
namespace buddy {
void registerLowerImgPass();
} // namespace buddy
} // namespace mlir

int main(int argc, char **argv) {
  // Register all MLIR passes and image dialect passes.
  mlir::registerAllPasses();
  mlir::buddy::registerLowerImgPass();
  // Register image dialect.
  mlir::DialectRegistry registry;
  registry.insert<buddy::img::ImgDialect>();
  // Register all MLIR dialect.
  registerAllDialects(registry);

  return failed(mlir::MlirOptMain(
      argc, argv, "Digital image processing optimizer driver\n", registry));
}
