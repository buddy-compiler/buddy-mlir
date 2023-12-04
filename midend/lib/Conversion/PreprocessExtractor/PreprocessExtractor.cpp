//===- PreprocessExtractor.cpp - ------------------------------------------===//
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
// This pass implements extract the operate only depend arg0 in BuddyLlama.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include <cstdint>
#include <functional>
#include <vector>

using namespace mlir;
using namespace MemoryEffects;

//===----------------------------------------------------------------------===//
// PreprocessExtractorPass
//===----------------------------------------------------------------------===//
namespace {
class PreprocessExtractorPass
    : public PassWrapper<PreprocessExtractorPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PreprocessExtractorPass)
  StringRef getArgument() const final { return "preprocess-extractor"; }
  StringRef getDescription() const final {
    return "Extract the operate only depend arg0 in BuddyLlama.";
  }
  PreprocessExtractorPass() = default;
  PreprocessExtractorPass(const PreprocessExtractorPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }
};
} // namespace

const auto isOpOperandsLegal =
    std::function<bool(Operation *, llvm::SmallPtrSet<Value, 16> &)>(
        [](Operation *userOp,
           SmallPtrSet<Value, 16> &funcAttrArgMemrefViewFlow) -> auto {
          std::vector<SmallVector<MemoryEffects::EffectInstance, 1>>
              effectsOfOperands;
          auto memrefEffects =
              llvm::dyn_cast<mlir::MemoryEffectOpInterface>(userOp);
          std::vector<bool> operandsHaveWriteEffect;
          auto operands = userOp->getOpOperands();

          // Iterate over operands to collect memory effects.
          for (auto &operand : operands) {
            effectsOfOperands.push_back(
                SmallVector<MemoryEffects::EffectInstance, 1>{});
            memrefEffects.getEffectsOnValue(operand.get(),
                                            effectsOfOperands.back());
            operandsHaveWriteEffect.push_back(false);
          }

          auto operandsPtr = operands.begin();

          // Analyze the effects on each operand.
          for (auto effects : effectsOfOperands) {
            auto currentOperandHasWriteEffect = false;
            auto currentOperandHasReadEffect = false;

            // Check effects on current operand.
            for (auto effect : effects) {
              if (isa<MemoryEffects::Write>(effect.getEffect())) {
                currentOperandHasWriteEffect = true;
                int usersWriteCount = 0;
                for (auto user : operandsPtr->get().getUsers()) {
                  SmallVector<MemoryEffects::EffectInstance, 1> outputEffects;
                  llvm::dyn_cast<mlir::MemoryEffectOpInterface>(user)
                      .getEffectsOnValue(operandsPtr->get(), outputEffects);
                  for (auto outputEffect : outputEffects) {
                    if (isa<MemoryEffects::Write>(outputEffect.getEffect())) {
                      usersWriteCount++;
                    }
                  }
                }
                if (usersWriteCount > 1) {
                  return false;
                }
              }
              if (isa<MemoryEffects::Read>(effect.getEffect())) {
                currentOperandHasReadEffect = true;
              }
            }

            // Check for illegal read without write on MemRefType
            if (currentOperandHasReadEffect and
                not currentOperandHasWriteEffect and
                operandsPtr->get().getType().isa<MemRefType>() and
                not funcAttrArgMemrefViewFlow.contains(operandsPtr->get())) {
              return false;
            }
            operandsPtr++;
          }
          return true;
        });

void PreprocessExtractorPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<memref::MemRefDialect, tensor::TensorDialect,
                         scf::SCFDialect>();

  auto opStack = std::vector<Operation *>();

  module->walk([&](func::FuncOp func) {
    auto block = func.getBlocks().begin();
    if (block->getNumArguments() > 0) {
      auto funcMemrefViewFlow = BufferViewFlowAnalysis(func);
      auto funcAttrArgMemrefViewFlow =
          funcMemrefViewFlow.resolve(block->getArguments().front());

      for (auto reshapeOp : funcAttrArgMemrefViewFlow) {
        auto users = reshapeOp.getUsers();

        for (auto user : users) {
          auto resultCounts = user->getNumResults();
          if (isOpOperandsLegal(user, funcAttrArgMemrefViewFlow)) {
            if (resultCounts > 0) {
              for (auto opResult : user->getOpResults()) {
                if (not opResult.getType().isa<MemRefType>() or
                    funcAttrArgMemrefViewFlow.contains(opResult)) {
                  goto skip;
                }
              }
            }
            opStack.push_back(user);
          }
        skip:
          continue;
        }
      }
    }

    OpBuilder builder = OpBuilder(func.getContext());

    for (auto op : opStack) {
      auto op0 = op->getOperand(0);
      auto op1 = op->getOperand(1);
      auto memrefTypeOfOp0 = op0.getType().cast<MemRefType>();
      auto memrefTypeOfOp1 = op1.getType().cast<MemRefType>();
      if (memrefTypeOfOp0.getShape()[0] == memrefTypeOfOp1.getShape()[1] and
          memrefTypeOfOp0.getShape()[1] == memrefTypeOfOp1.getShape()[0] and
          memrefTypeOfOp0.getRank() == 2 and memrefTypeOfOp1.getRank() == 2) {
        builder.setInsertionPointAfter(op);

        auto shape = op0.getType().dyn_cast<MemRefType>().getShape();
        auto layout = op0.getType()
                          .dyn_cast<MemRefType>()
                          .getLayout()
                          .cast<StridedLayoutAttr>();
        auto transposedShapeMemref = builder.create<memref::ExpandShapeOp>(
            op->getLoc(),
            MemRefType::get(
                ArrayRef<int64_t>{shape[1], shape[0]}, builder.getF32Type(),
                StridedLayoutAttr::get(func->getContext(), layout.getOffset(),
                                       ArrayRef<int64_t>{shape[0], 1})),
            op0.getDefiningOp()->getOpOperand(0).get(),
            ArrayRef{ReassociationIndices{0, 1}});

        for (auto user : op1.getUsers()) {
          if (user == op) {
            continue;
          }
          SmallVector<MemoryEffects::EffectInstance, 1> effects;
          auto memrefEffects =
              llvm::dyn_cast<mlir::MemoryEffectOpInterface>(user);
          memrefEffects.getEffectsOnValue(op1, effects);
          for (auto &effect : effects) {
            if (isa<MemoryEffects::Write>(effect.getEffect())) {
              goto out;
            }
          }
          for (uint i = 0; i < user->getNumOperands(); i++) {
            if (user->getOperand(i) == op1) {
              user->setOperand(i, transposedShapeMemref);
            }
          }
        out:
          continue;
        }

        auto zeroIndex = builder.create<arith::ConstantOp>(
            op->getLoc(), builder.getIndexAttr(0));
        auto stepOneIndex = builder.create<arith::ConstantOp>(
            op->getLoc(), builder.getIndexAttr(1));
        auto ubOuterLoop = builder.create<arith::ConstantOp>(
            op->getLoc(), builder.getIndexAttr(memrefTypeOfOp1.getShape()[0]));
        auto ubInnerLoop = builder.create<arith::ConstantOp>(
            op->getLoc(), builder.getIndexAttr(memrefTypeOfOp1.getShape()[1]));
        auto loop = scf::buildLoopNest(
            builder, op->getLoc(), {zeroIndex, zeroIndex},
            {ubOuterLoop, ubInnerLoop}, {stepOneIndex, stepOneIndex},
            [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
              auto loadValue = builder.create<memref::LoadOp>(
                  loc, op1, ValueRange{ivRange[0], ivRange[1]});
              builder.create<memref::StoreOp>(
                  loc, loadValue, op0, ValueRange{ivRange[0], ivRange[1]});
            });
      }
    }
  });
}

namespace mlir {
namespace buddy {
void registerPreprocessExtractorPass() {
  PassRegistration<PreprocessExtractorPass>();
}
} // namespace buddy
} // namespace mlir
