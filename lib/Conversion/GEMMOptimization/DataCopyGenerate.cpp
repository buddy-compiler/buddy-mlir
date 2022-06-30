//===- DataCopyGenerate.cpp - Generate Data Copy For GEMM ------===//
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
// This file implements Data Copy Generate algorithm.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <ostream>
#include <string>
#include <type_traits>

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// DataCopyGeneratePass
//===----------------------------------------------------------------------===//

namespace {
class DataCopyGeneratePass
    : public PassWrapper<DataCopyGeneratePass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DataCopyGeneratePass)
  StringRef getArgument() const final { return "gemm-data-copy-generate"; }
  StringRef getDescription() const final {
    return "Generate data copy for GEMM.";
  }
  DataCopyGeneratePass() = default;
  DataCopyGeneratePass(const DataCopyGeneratePass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, func::FuncDialect>();
  }
};
} // end anonymous namespace.

void DataCopyGeneratePass::runOnOperation() {
  auto func = getOperation();
  AffineForOp innermoustFor;

  func.walk<mlir::WalkOrder::PreOrder>([&](AffineForOp forOp){
    innermoustFor = forOp;
    return;
  });

  func.walk([&](AffineForOp forOp) {
    if (getNestingDepth(forOp) == 2) {
      DenseSet<Operation *> denseSet;
      (void)affineDataCopyGenerate(
          forOp, AffineCopyOptions(),
          llvm::Optional<Value>(), denseSet);
    }
    return;
  });
  (void)loopUnrollJamByFactor(innermoustFor, 4);
}

namespace mlir {
namespace buddy {
void registerDataCopyGeneratePass() {
  PassRegistration<DataCopyGeneratePass>();
}
} // namespace buddy
} // namespace mlir
