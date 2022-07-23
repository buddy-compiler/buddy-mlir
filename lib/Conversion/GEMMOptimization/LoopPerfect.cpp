//===- LoopPerfect.cpp - Make loop perfect ------===//
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
// This file implements Loop perfect algorithm.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"
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
// LoopPerfectPass
//===----------------------------------------------------------------------===//

namespace {
class LoopPerfectPass
    : public PassWrapper<LoopPerfectPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoopPerfectPass)
  StringRef getArgument() const final { return "gemm-loop-perfect"; }
  StringRef getDescription() const final {
    return "Generate data copy for GEMM.";
  }
  LoopPerfectPass() = default;
  LoopPerfectPass(const LoopPerfectPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, func::FuncDialect>();
  }
};
} // end anonymous namespace.

void LoopPerfectPass::runOnOperation() {
  auto func = getOperation();

  AffineForOp innermostForOp;
  func.walk([&](AffineForOp op){
    innermostForOp = op;
    return WalkResult::interrupt();
  });

  llvm::SmallVector<Operation*> v;
  func.walk<WalkOrder::PreOrder>([&](AffineForOp forOp){
    if(innermostForOp != forOp){
        for(auto &op : forOp.getLoopBody().getOps()){
            if(!llvm::isa<AffineForOp, AffineYieldOp>(op)){
                v.push_back(&op);
            }
        }
    }else{
        OpBuilder b(forOp.getLoopBody());
        for(auto moveNeededOp : v){
            auto newOp = moveNeededOp->clone();
            b.insert(newOp);
            moveNeededOp->replaceAllUsesWith(newOp);
            moveNeededOp->erase();
        }
    }
  });
}

namespace mlir {
namespace buddy {
void registerLoopPerfectPass() {
  PassRegistration<LoopPerfectPass>();
}
} // namespace buddy
} // namespace mlir
