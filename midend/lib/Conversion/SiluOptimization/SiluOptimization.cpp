//====- SiluOptimization.cpp - Silu Optimization Pass ---------------------===//
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
// This file implements the pass that vectorizes the linalg.generic representing
// SiLU.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

class SiLUVectorizePattern : public ConversionPattern {
public:
  explicit SiLUVectorizePattern(MLIRContext *context, int64_t vectorSizeParam)
      : ConversionPattern(linalg::GenericOp::getOperationName(), 1, context) {
    vectorSize = vectorSizeParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    linalg::GenericOp sigmoidOp = cast<linalg::GenericOp>(op);

    //--------------sigmoid OP--------
    //  Check input/output
    if (sigmoidOp.getNumDpsInputs() != 1 || sigmoidOp.getNumDpsInits() != 1) {
      return failure();
    }

    // Check the body of the op for sigmoid computation.
    // The IR should be: negf, exp, addf, divf, yield.
    Block &block = sigmoidOp.getRegion().front();
    if (block.getOperations().size() != 5) { // negf, exp, addf, divf, yield
      return failure();
    }

    Operation &negfOp = block.getOperations().front();
    Operation &yieldOp = block.getOperations().back();

    // Check the type of the two operations.
    if (!isa<arith::NegFOp>(negfOp) || !isa<linalg::YieldOp>(yieldOp)) {
      return failure();
    }

    //-----------Find the consumer mul operation.------------------------------
    // The result of the sigmoid op must be used by another linalg.generic op.
    Value outputBuffer = sigmoidOp.getDpsInitOperand(0)->get();

    // Iterate over all uses to find a suitable consumer op.
    linalg::GenericOp mulOp = nullptr;

    for (auto &use : outputBuffer.getUses()) {
      Operation *user = use.getOwner();

      // It must be a linalg.generic, and the buffer must be an input operand
      // (i.e., ins()).
      auto linalgOp = dyn_cast<linalg::GenericOp>(user);
      if (!linalgOp)
        continue;

      bool foundInInput = false;
      for (OpOperand *input : linalgOp.getDpsInputOperands()) {
        if (input->get() == outputBuffer) {
          foundInInput = true;
          break;
        }
      }
      if (!foundInInput)
        continue;

      // Check if it contains an arith.mulf operation inside.
      for (auto &nestedOp : linalgOp.getRegion().front()) {
        if (isa<arith::MulFOp>(nestedOp)) {
          mulOp = linalgOp;
          break;
        }
      }

      if (mulOp)
        break;
    }

    if (!mulOp) {
      llvm::errs() << "Didn't find a consumer linalg.generic using sigmoid "
                      "output with mulf.\n";
      return failure();
    }

    // Set the insertion point before the mulOp. This ensures that the new
    // affine loop is inserted at a point that is dominated by the allocation of
    // the output buffer. rewriter.setInsertionPoint(mulOp);

    // Now we have matched the silu pattern: sigmoid followed by a mul.
    // The rewrite logic will be applied to the sigmoidOp, and the mulOp will be
    // erased.

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(mulOp);
    Location loc = sigmoidOp.getLoc();
    Value input = sigmoidOp.getDpsInputOperand(0)->get();
    // The final output buffer comes from the mulOp.
    Value output = mulOp.getDpsInitOperand(0)->get();

    auto inputMemRefType = input.getType().cast<MemRefType>();
    Type elementType = inputMemRefType.getElementType();
    VectorType vectorType = VectorType::get({vectorSize}, elementType);

    // Define constants.
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value cst1f =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(1.0));
    Value vec1f = rewriter.create<vector::BroadcastOp>(loc, vectorType, cst1f);
    Value cst0f =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));

    // Get dimensions.
    Value d0 = rewriter.create<memref::DimOp>(loc, input, 0);
    Value d1 = rewriter.create<memref::DimOp>(loc, input, 1);
    Value d2 = rewriter.create<memref::DimOp>(loc, input, 2);

    // Create loop nest.
    AffineMap map = rewriter.getDimIdentityMap();
    affine::AffineForOp iLoop = rewriter.create<affine::AffineForOp>(
        loc, ValueRange{c0}, map, ValueRange{d0}, map);
    rewriter.setInsertionPointToStart(iLoop.getBody());
    Value iv_i = iLoop.getInductionVar();

    affine::AffineForOp jLoop = rewriter.create<affine::AffineForOp>(
        loc, ValueRange{c0}, map, ValueRange{d1}, map);
    rewriter.setInsertionPointToStart(jLoop.getBody());
    Value iv_j = jLoop.getInductionVar();

    affine::AffineForOp kLoop = rewriter.create<affine::AffineForOp>(
        loc, ValueRange{c0}, map, ValueRange{d2}, map, vectorSize);
    rewriter.setInsertionPointToStart(kLoop.getBody());
    Value iv_k = kLoop.getInductionVar();

    // --- Process Vector ---
    Value x_vec = rewriter.create<vector::TransferReadOp>(
        loc, vectorType, input, ValueRange{iv_i, iv_j, iv_k}, cst0f);
    Value neg_x_vec = rewriter.create<arith::NegFOp>(loc, x_vec);
    Value exp_neg_x_vec = rewriter.create<math::ExpOp>(loc, neg_x_vec);
    Value one_plus_exp_vec =
        rewriter.create<arith::AddFOp>(loc, vec1f, exp_neg_x_vec);
    Value sigmoid_x_vec =
        rewriter.create<arith::DivFOp>(loc, vec1f, one_plus_exp_vec);
    Value silu_vec = rewriter.create<arith::MulFOp>(loc, x_vec, sigmoid_x_vec);
    rewriter.create<vector::TransferWriteOp>(loc, silu_vec, output,
                                             ValueRange{iv_i, iv_j, iv_k});

    // Replace the original mulOp with the result from our new computation.
    // The 'output' buffer now holds the final result. `replaceOp` will
    // replace all uses of mulOp's results with `output` and then erase mulOp.
    rewriter.eraseOp(mulOp);
    rewriter.eraseOp(sigmoidOp);

    return success();
  }

private:
  int64_t vectorSize;
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class SiluOptimizationPass
    : public PassWrapper<SiluOptimizationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SiluOptimizationPass)
  StringRef getArgument() const final { return "silu-optimization"; }
  StringRef getDescription() const final {
    return "Vectorize linalg.generic representing SiLU.";
  }
  SiluOptimizationPass() = default;
  SiluOptimizationPass(const SiluOptimizationPass &) {}
  explicit SiluOptimizationPass(int64_t vectorSizeParam) {
    vectorSize = vectorSizeParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, affine::AffineDialect,
                    vector::VectorDialect, math::MathDialect, scf::SCFDialect,
                    arith::ArithDialect, memref::MemRefDialect>();
  }

  Option<int64_t> vectorSize{*this, "vector-size",
                             llvm::cl::desc("Vector size for SiLU."),
                             llvm::cl::init(8)};
};

void SiluOptimizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       memref::MemRefDialect, vector::VectorDialect,
                       func::FuncDialect, math::MathDialect, scf::SCFDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp>();

  // We will manually mark linalg.generic as illegal if it is part of a SiLU
  // pattern. The pattern itself will handle the legality checks and
  // replacements. Therefore, we don't need to addIllegalOp<linalg::GenericOp>()
  // here.

  RewritePatternSet patterns(context);
  patterns.add<SiLUVectorizePattern>(context, vectorSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace
namespace mlir {
namespace buddy {
void registerSiluOptimizationPass() {
  PassRegistration<SiluOptimizationPass>();
}
} // namespace buddy
} // namespace mlir