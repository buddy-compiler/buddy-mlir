//===- OutlineLoopNests.cpp - One function per top-level loop nest --------===//
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
// This pass moves every top-level scf.for and scf.parallel of a function into
// a private function of its own, called in its place. A whole model lowered
// into one function gives LLVM one function of millions of instructions, and
// LLVM's code generation grows faster than linearly with the size of a
// function; one function per loop nest keeps it linear.
//
// Values the loop nest reads from the function become arguments of the new
// function; index constants are cloned into it. The new functions are
// noinline: LLVM would inline every function called once back into the
// function, which then only calls the loop nests in order.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;

namespace {

class OutlineLoopNestsPass
    : public PassWrapper<OutlineLoopNestsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OutlineLoopNestsPass)
  StringRef getArgument() const final { return "outline-loop-nests"; }
  StringRef getDescription() const final {
    return "Move every top-level loop nest of a function into a function of "
           "its own.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    SmallVector<func::FuncOp> functions(getOperation().getOps<func::FuncOp>());
    for (func::FuncOp function : functions) {
      if (function.isExternal())
        continue;
      SmallVector<Operation *> nests;
      for (Operation &op : function.getBody().getOps())
        if (isa<scf::ForOp, scf::ParallelOp>(op))
          nests.push_back(&op);
      for (auto [index, nest] : llvm::enumerate(nests)) {
        SetVector<Value> captures;
        getUsedValuesDefinedAbove(nest->getRegions(), captures);
        captures.insert(nest->operand_begin(), nest->operand_end());
        std::string name = (function.getName() + "_nest" + Twine(index)).str();
        ValueRange inputs(captures.getArrayRef());
        rewriter.setInsertionPoint(function);
        auto outlined = func::FuncOp::create(
            rewriter, nest->getLoc(), name,
            rewriter.getFunctionType(inputs.getTypes(),
                                     nest->getResultTypes()));
        outlined.setPrivate();
        outlined->setAttr("llvm.passthrough",
                          rewriter.getStrArrayAttr({"noinline"}));
        rewriter.setInsertionPointToStart(outlined.addEntryBlock());
        IRMapping mapping;
        for (auto [value, argument] :
             llvm::zip(inputs, outlined.getArguments())) {
          Value replacement = argument;
          if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>()) {
            rewriter.setInsertionPointToStart(&outlined.getBody().front());
            replacement = rewriter.clone(*constant)->getResult(0);
          }
          mapping.map(value, replacement);
        }
        // Clone only this nest; scanning each capture's global use list
        // repeatedly would make a function with many nests quadratic.
        rewriter.setInsertionPointToEnd(&outlined.getBody().front());
        Operation *body = rewriter.clone(*nest, mapping);
        func::ReturnOp::create(rewriter, nest->getLoc(), body->getResults());
        rewriter.setInsertionPoint(nest);
        auto call =
            func::CallOp::create(rewriter, nest->getLoc(), outlined, inputs);
        rewriter.replaceOp(nest, call.getResults());
      }
    }
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerOutlineLoopNestsPass() {
  PassRegistration<OutlineLoopNestsPass>();
}
} // namespace buddy
} // namespace mlir
