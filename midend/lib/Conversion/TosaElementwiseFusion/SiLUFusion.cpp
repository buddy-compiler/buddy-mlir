//===- SiLUFusion.cpp ----------------------------------------------------===//
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
// This file implements the SiLU fusion pass that automatically detects and
// fuses tosa.sigmoid + tosa.mul patterns into a single linalg.generic
// operation.
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;

//===----------------------------------------------------------------------===//
// SiLU Fusion Pattern
//===----------------------------------------------------------------------===//

namespace {

class SiLUFusionPattern : public OpRewritePattern<tosa::MulOp> {
public:
  using OpRewritePattern<tosa::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MulOp mulOp,
                                PatternRewriter &rewriter) const override {
    auto loc = mulOp.getLoc();

    Value lhs = mulOp.getInput1();
    Value rhs = mulOp.getInput2();

    // Check for pattern: x * sigmoid(x) or sigmoid(x) * x
    tosa::SigmoidOp sigmoidOp = nullptr;
    Value inputValue = nullptr;

    if (auto lhsSigmoid = lhs.getDefiningOp<tosa::SigmoidOp>()) {
      if (lhsSigmoid.getInput() == rhs) {
        sigmoidOp = lhsSigmoid;
        inputValue = rhs;
      }
    } else if (auto rhsSigmoid = rhs.getDefiningOp<tosa::SigmoidOp>()) {
      if (rhsSigmoid.getInput() == lhs) {
        sigmoidOp = rhsSigmoid;
        inputValue = lhs;
      }
    }

    if (!sigmoidOp || !inputValue) {
      return failure();
    }

    // Check if sigmoid has only one use (the multiply operation)
    if (!sigmoidOp.getResult().hasOneUse()) {
      return failure();
    }

    // Get tensor type information
    auto inputType = mlir::cast<RankedTensorType>(inputValue.getType());
    auto outputType = mlir::cast<RankedTensorType>(mulOp.getResult().getType());
    Type elementType = inputType.getElementType();

    // Create affine maps for linalg.generic
    int64_t rank = inputType.getRank();
    SmallVector<AffineExpr> exprs;
    for (int64_t i = 0; i < rank; ++i) {
      exprs.push_back(rewriter.getAffineDimExpr(i));
    }
    AffineMap identityMap =
        AffineMap::get(rank, 0, exprs, rewriter.getContext());
    SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};

    // Create iterator types (all parallel)
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    // Create empty tensor for output
    tensor::EmptyOp emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, outputType.getShape(), elementType);

    // Create fused SiLU operation using linalg.generic
    linalg::GenericOp fusedOp = rewriter.create<linalg::GenericOp>(
        loc, outputType, ValueRange{inputValue}, ValueRange{emptyTensor},
        indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value input = args[0];

          // SiLU: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
          Value one = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(elementType, 1.0));
          Value negInput = b.create<arith::NegFOp>(loc, input);
          Value expNegInput = b.create<math::ExpOp>(loc, negInput);
          Value onePlusExp = b.create<arith::AddFOp>(loc, one, expNegInput);
          Value sigmoid = b.create<arith::DivFOp>(loc, one, onePlusExp);
          Value silu = b.create<arith::MulFOp>(loc, input, sigmoid);

          b.create<linalg::YieldOp>(loc, silu);
        });

    // Replace the original multiply operation
    rewriter.replaceOp(mulOp, fusedOp.getResult(0));

    // Erase the sigmoid operation (we already verified it has only one use)
    rewriter.eraseOp(sigmoidOp);

    return success();
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// SiLU Fusion Pass
//===----------------------------------------------------------------------===//

namespace {
class SiLUFusionPass
    : public PassWrapper<SiLUFusionPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SiLUFusionPass)

  StringRef getArgument() const final { return "silu-fusion"; }
  StringRef getDescription() const final {
    return "Fuse TOSA sigmoid and multiply operations into SiLU activation.";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<SiLUFusionPattern>(context);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace buddy {
void registerSiLUFusionPass() { PassRegistration<SiLUFusionPass>(); }
} // namespace buddy
} // namespace mlir
