//===- DIP-opt.cpp ----------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// DIP optimizer driver.
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

#include "DIP/DIPDialect.h"
#include "DIP/DIPOps.h"

namespace mlir {
namespace Buddy {
void registerLowerDIPPass();
} // namespace Buddy
} // namespace mlir

int main(int argc, char **argv) {
  // Register all MLIR passes and DIP dialect passes.
  mlir::registerAllPasses();
  mlir::Buddy::registerLowerDIPPass();
  // Register DIP dialect.
  mlir::DialectRegistry registry;
  registry.insert<Buddy::DIP::DIPDialect>();
  // Register all MLIR dialect.
  registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "DIP optimizer driver\n", registry));
}
