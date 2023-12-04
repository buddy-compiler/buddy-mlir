//===- CopyRemoval.cpp - Remove redundant copies --------------------------===//
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
// This pass implements remove the redundant copy operations.
// Inspired by https://reviews.llvm.org/D117673
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;
using namespace MemoryEffects;

//===----------------------------------------------------------------------===//
// CopyRemovalPass
//===----------------------------------------------------------------------===//
namespace {
class CopyRemovalPass
    : public PassWrapper<CopyRemovalPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CopyRemovalPass)
  StringRef getArgument() const final { return "copy-removal"; }
  StringRef getDescription() const final {
    return "Remove rebudant memref.copy";
  }
  CopyRemovalPass() = default;
  CopyRemovalPass(const CopyRemovalPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }
};
} // namespace

void CopyRemovalPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<memref::MemRefDialect, tensor::TensorDialect>();

  module->walk([&](memref::CopyOp op) {
    auto ops = op->getOperands();
    if (ops[0]
            .getType()
            .dyn_cast<MemRefType>()
            .getLayout()
            .isa<StridedLayoutAttr>()) {
      auto ub = ops[1].getType().cast<MemRefType>().getRank() - 1;
      for (int64_t i = 0; i < ub; i++) {
        if (not(ops[0]
                    .getType()
                    .dyn_cast<MemRefType>()
                    .getLayout()
                    .dyn_cast<StridedLayoutAttr>()
                    .getStrides()[i] ==
                ops[1].getType().cast<MemRefType>().getDimSize(ub - i))) {
          goto skip;
        }
      }
      if (not(ops[0]
                  .getType()
                  .dyn_cast<MemRefType>()
                  .getLayout()
                  .dyn_cast<StridedLayoutAttr>()
                  .getStrides()[ub] == 1)) {
        goto skip;
      }
      if (not ops[1].getUsers().empty()) {
        llvm::SmallVector<Operation *, 32> users(ops[1].getUsers().begin(),
                                                 ops[1].getUsers().end());
        while (not users.empty()) {
          auto *user = users.pop_back_val();
          if (user == op) {
            continue;
          }
          SmallVector<MemoryEffects::EffectInstance, 1> effects;
          auto memrefEffects =
              llvm::dyn_cast<mlir::MemoryEffectOpInterface>(user);
          memrefEffects.getEffectsOnValue(ops[1], effects);
          for (auto &effect : effects) {
            if (isa<MemoryEffects::Write>(effect.getEffect())) {
              goto skip;
            }
          }
          for (uint i = 0; i < user->getNumOperands(); i++) {
            if (user->getOperand(i) == ops[1]) {
              user->setOperand(i, ops[0]);
            }
          }
        }
        op->erase();
      }
    skip:
      return;
    }
  });
}

namespace mlir {
namespace buddy {
void registerCopyRemovalPass() { PassRegistration<CopyRemovalPass>(); }
} // namespace buddy
} // namespace mlir
