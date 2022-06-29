//===- LoopOrderChange.cpp - Change GEMM Loop order ------===//
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
// This file implements GEMM Loop Order Change algorithm.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
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
// LoopOrderChangePass
//===----------------------------------------------------------------------===//

namespace {
class LoopOrderChangePass
    : public PassWrapper<LoopOrderChangePass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoopOrderChangePass)
  StringRef getArgument() const final { return "loop-order-change"; }
  StringRef getDescription() const final {
    return "change loop order, for now only test GEMM case.";
  }
  LoopOrderChangePass() = default;
  LoopOrderChangePass(const LoopOrderChangePass &) {}
  explicit LoopOrderChangePass(ArrayRef<int64_t> order) {
    newLoopOrder = order;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, func::FuncDialect>();
  }

  ListOption<int64_t> newLoopOrder{*this, "new-order-list",
                                   llvm::cl::desc("New order list."),
                                   llvm::cl::ZeroOrMore};
};
} // end anonymous namespace.

void LoopOrderChangePass::runOnOperation() {
  auto func = getOperation();

  AffineForOp outmostForOp;
  func->walk([&](AffineForOp op) {
    outmostForOp = op;
    return;
  });

  if (getNestingDepth(outmostForOp) != 0)
    return;

  SmallVector<AffineForOp, 3> loopList;

  // if we use this way to collect loopOp, means there have no extra op between
  // loops.
  getPerfectlyNestedLoops(loopList, outmostForOp);

  int loopSize = loopList.size();


  SmallVector<int, 3> loopIdx;
  for(int i = 0; i < loopSize; ++ i) loopIdx.push_back(i);

  auto swap = [&](int i, int j){
        interchangeLoops(loopList[i], loopList[j]);
        std::swap(loopList[i], loopList[j]);
        std::swap(loopIdx[i], loopIdx[j]);
  };

  for(int i = 0; i < loopSize; ++ i){
      int j = newLoopOrder[i];
      if(i != j){
        int idxA = i;
        int idxB = llvm::find(loopIdx, j) - loopIdx.begin();

        if(idxA > idxB) std::swap(idxA, idxB);

        for(int p = idxA; p + 1 <= idxB; ++ p){
            swap(p, p + 1);
        }
        for(int p = idxB - 1; p - 1 >= idxA; -- p){
            swap(p - 1, p);
        }
      }
  }

}

namespace mlir {
namespace buddy {
void registerLoopOrderChangePass() { PassRegistration<LoopOrderChangePass>(); }
} // namespace buddy
} // namespace mlir
