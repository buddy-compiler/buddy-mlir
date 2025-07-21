//===----------- InitAll.cpp - Register all dialects and passes -----------===//
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

#include "buddy-mlir-c/InitAll.h"

#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"

#include "Dialect/Bud/BudDialect.h"
#include "Dialect/DAP/DAPDialect.h"
#include "Dialect/DIP/DIPDialect.h"
#include "Dialect/Gemmini/GemminiDialect.h"
#include "Dialect/RVV/RVVDialect.h"
#include "Dialect/VectorExp/VectorExpDialect.h"

namespace mlir {
namespace buddy {
void registerConvOptimizePass();
void registerConvVectorizationPass();
void registerPointwiseConvToGemmPass();
void registerPoolingVectorizationPass();
void registerLowerBudPass();
void registerLowerDAPPass();
void registerLowerDIPPass();
void registerLowerGemminiPass();
void registerLowerLinalgToGemminiPass();
void registerLowerRVVPass();
void registerLowerVectorExpPass();
void registerBatchMatMulOptimizePass();
void registerMatMulOptimizePass();
void registerMatMulParallelVectorizationPass();
void registerMatMulVectorizationPass();
void registerTransposeOptimizationPass();
} // namespace buddy
} // namespace mlir

void mlir::buddy::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<::buddy::bud::BudDialect>();
  registry.insert<::buddy::dap::DAPDialect>();
  registry.insert<::buddy::dip::DIPDialect>();
  registry.insert<::buddy::gemmini::GemminiDialect>();
  registry.insert<::buddy::rvv::RVVDialect>();
  registry.insert<::buddy::vector_exp::VectorExpDialect>();
}

void mlir::buddy::registerAllPasses() {
  mlir::buddy::registerConvOptimizePass();
  mlir::buddy::registerConvVectorizationPass();
  mlir::buddy::registerPointwiseConvToGemmPass();
  mlir::buddy::registerPoolingVectorizationPass();
  mlir::buddy::registerLowerBudPass();
  mlir::buddy::registerLowerDAPPass();
  mlir::buddy::registerLowerDIPPass();
  mlir::buddy::registerLowerGemminiPass();
  mlir::buddy::registerLowerLinalgToGemminiPass();
  mlir::buddy::registerLowerRVVPass();
  mlir::buddy::registerLowerVectorExpPass();
  mlir::buddy::registerBatchMatMulOptimizePass();
  mlir::buddy::registerMatMulOptimizePass();
  mlir::buddy::registerMatMulParallelVectorizationPass();
  mlir::buddy::registerMatMulVectorizationPass();
  mlir::buddy::registerTransposeOptimizationPass();
}
