//====- buddy-opt.cpp - The driver of buddy-mlir --------------------------===//
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
// This file is the dialects and optimization driver of buddy-mlir project.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
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
#include "DAP/DAPDialect.h"
#include "DAP/DAPOps.h"
#include "DIP/DIPDialect.h"
#include "DIP/DIPOps.h"
#include "RVV/RVVDialect.h"
#include "VectorExp/VectorExpDialect.h"
#include "VectorExp/VectorExpOps.h"
#include "Gemmini/GemminiDialect.h"
#include "Gemmini/GemminiOps.h"

namespace mlir {
namespace buddy {
void registerConvVectorizationPass();
void registerPointwiseConvToGemmPass();
void registerPoolingVectorizationPass();
void registerLowerBudPass();
void registerLowerDIPPass();
void registerLowerDAPPass();
void registerLowerRVVPass();
void registerBatchMatMulOptimizePass();
void registerMatMulOptimizePass();
void registerMatMulVectorizationPass();

void registerConvOptimizePass();
void registerLowerVectorExpPass();
void registerLowerGemminiPass();
void registerLowerLinalgToGemminiPass();
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
  mlir::buddy::registerLowerDAPPass();
  mlir::buddy::registerLowerRVVPass();
  mlir::buddy::registerLowerVectorExpPass();
  mlir::buddy::registerLowerGemminiPass();
  mlir::buddy::registerLowerLinalgToGemminiPass();

  // Register Several Optimize Pass.
  mlir::buddy::registerMatMulOptimizePass();
  mlir::buddy::registerMatMulVectorizationPass();
  mlir::buddy::registerBatchMatMulOptimizePass();
  mlir::buddy::registerConvOptimizePass();

  mlir::DialectRegistry registry;
  // Register all MLIR core dialects.
  registerAllDialects(registry);
  // Register dialects in buddy-mlir project.
  // clang-format off
  registry.insert<buddy::bud::BudDialect,
                  buddy::dip::DIPDialect,
                  buddy::dap::DAPDialect,
                  buddy::rvv::RVVDialect,
                  buddy::vector_exp::VectorExpDialect,
                  buddy::gemmini::GemminiDialect>();
  // clang-format on

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "buddy-mlir optimizer driver", registry));
}
