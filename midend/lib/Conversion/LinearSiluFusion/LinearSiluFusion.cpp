//===- LinearSiluFusion.cpp - Linear+SiLU Fusion Pass --------------------===//
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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tensor;
using namespace mlir::arith;
using namespace mlir::linalg;
using namespace mlir::bufferization;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Check if a linalg.generic operation implements sigmoid function
bool isSigmoidOp(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1) {
    return false;
  }

  Block &block = op.getRegion().front();
  auto it = block.getOperations().begin();

  // Check: negf, exp, addf, divf, yield
  if (!isa<arith::NegFOp>(*it++)) return false;
  if (!isa<math::ExpOp>(*it++)) return false;
  if (!isa<arith::AddFOp>(*it++)) return false;
  if (!isa<arith::DivFOp>(*it++)) return false;
  if (!isa<linalg::YieldOp>(*it)) return false;

  return true;
}

/// Check if a linalg.generic operation implements simple multiplication
bool isMulOp(linalg::GenericOp op) {
  for (auto &nestedOp : op.getRegion().front()) {
    if (isa<arith::MulFOp>(nestedOp)) {
      return true;
    }
  }
  return false;
}

/// Find the consumer linalg.generic operation that uses the given value
/// and contains a multiplication operation
linalg::GenericOp findMulConsumer(Value value) {
  for (auto &use : value.getUses()) {
    Operation *user = use.getOwner();
    auto linalgOp = dyn_cast<linalg::GenericOp>(user);
    if (!linalgOp) {
      continue;
    }

    // Check if the value is used as an input operand
    bool foundInInput = false;
    for (OpOperand *input : linalgOp.getDpsInputOperands()) {
      if (input->get() == value) {
        foundInInput = true;
        break;
      }
    }

    if (!foundInInput) {
      continue;
    }

    // Check if it contains an arith.mulf operation inside
    if (isMulOp(linalgOp)) {
      return linalgOp;
    }
  }
  return nullptr;
}

/// Check if a linalg.generic operation implements simple addition
bool isAddOp(linalg::GenericOp op) {
  for (auto &nestedOp : op.getRegion().front()) {
    if (isa<arith::AddFOp>(nestedOp)) {
      return true;
    }
  }
  return false;
}
//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

class LinearSiluFusionPattern : public OpRewritePattern<linalg::BatchMatmulOp> {
public:
  explicit LinearSiluFusionPattern(MLIRContext *context)
      : OpRewritePattern<linalg::BatchMatmulOp>(context, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(linalg::BatchMatmulOp batchMatmulOp,
                  PatternRewriter &rewriter) const override {
    // Step 1: Find consumers of batch_matmul result (collapse_shape)
    Value batchMatmulResult = batchMatmulOp.getResult(0);
    tensor::CollapseShapeOp collapseShapeOp = nullptr;
    for (auto &use : batchMatmulResult.getUses()) {
      if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(use.getOwner())) {
        collapseShapeOp = collapseOp;
        break;
      }
    }
    if (!collapseShapeOp) {
      return failure();
    }

    // Step 2: Find consumers of collapse_shape output (add operation)
    Value collapseOutput = collapseShapeOp.getResult();
    linalg::GenericOp addOp = nullptr;
    for (auto &use : collapseOutput.getUses()) {
      if (auto linalgOp = dyn_cast<linalg::GenericOp>(use.getOwner())) {
        if (isAddOp(linalgOp)) {
          addOp = linalgOp;
          break;
        }
      }
    }
    if (!addOp) {
      return failure();
    }

    // Step 3: Find the bias expand_shape from add operation's first input
    Value addFirstInput = addOp.getDpsInputOperand(0)->get();
    tensor::ExpandShapeOp biasExpandOp = nullptr;
    if (auto expandOp = addFirstInput.getDefiningOp<tensor::ExpandShapeOp>()) {
      biasExpandOp = expandOp;
    }
    if (!biasExpandOp) {
      return failure();
    }

    // Step 4: Find the tensor.empty() operation that initializes the add op
    Value addInit = addOp.getDpsInitOperand(0)->get();
    auto addInitOp = addInit.getDefiningOp<tensor::EmptyOp>();
    if (!addInitOp) {
      return failure();
    }

    // Step 5: Find consumers of add result (expand_shape for sigmoid)
    Value addResult = addOp.getResult(0);
    tensor::ExpandShapeOp sigmoidExpandOp = nullptr;
    for (auto &use : addResult.getUses()) {
      if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(use.getOwner())) {
        sigmoidExpandOp = expandOp;
        break;
      }
    }
    if (!sigmoidExpandOp) {
      return failure();
    }

    // Step 6: Find the unique consumer of sigmoid expand_shape output (sigmoid)
    Value sigmoidExpandOutput = sigmoidExpandOp.getResult();

    if (std::distance(sigmoidExpandOutput.user_begin(), sigmoidExpandOutput.user_end()) != 2) {
      return failure();
    }

    // Identify sigmoid and mul operations
    linalg::GenericOp sigmoidOp = nullptr;
    linalg::GenericOp mulOp = nullptr;

    for (Operation *user : sigmoidExpandOutput.getUsers()) {
      auto genericUser = dyn_cast<linalg::GenericOp>(user);
      if (!genericUser) {
        return failure();
      }

      if (isSigmoidOp(genericUser)) {
        sigmoidOp = genericUser;
      } else if (isMulOp(genericUser)) {
        mulOp = genericUser;
      }
    }

    if (!sigmoidOp || !mulOp) {
      return failure();
    }

    // Verify mul operation inputs match x * sigmoid(x) pattern
    Value sigmoidResult = sigmoidOp->getResult(0);
    bool isCorrectSiLUMul = (mulOp->getOperand(0) == sigmoidExpandOutput && mulOp->getOperand(1) == sigmoidResult) ||
                           (mulOp->getOperand(0) == sigmoidResult && mulOp->getOperand(1) == sigmoidExpandOutput);

    if (!isCorrectSiLUMul) {
      return failure();
    }

    // Step 7: Find the tensor.empty() operation that initializes the sigmoid op
    Value sigmoidInit = sigmoidOp.getDpsInitOperand(0)->get();
    auto sigmoidInitOp = sigmoidInit.getDefiningOp<tensor::EmptyOp>();
    if (!sigmoidInitOp) {
      return failure();
    }

    // Step 8: Find the tensor.empty() operation that initializes the mul op
    Value mulInit = mulOp.getDpsInitOperand(0)->get();
    auto mulInitOp = mulInit.getDefiningOp<tensor::EmptyOp>();
    if (!mulInitOp) {
      return failure();
    }

    // =================================================================
    // Create fused linalg.generic implementation
    // =================================================================

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(addInitOp);
    Location loc = addInitOp.getLoc();
    MLIRContext *context = rewriter.getContext();

    Value inputA = batchMatmulOp.getDpsInputOperand(0)->get();
    Value inputB = batchMatmulOp.getDpsInputOperand(1)->get();
    Value biasInput = biasExpandOp.getResult();
    auto finalResultType = mulOp.getDpsInitOperand(0)->get().getType().cast<RankedTensorType>();

    // Create affine maps for the first operation (matmul)
    const int numDims = 4;
    AffineMap mapA = AffineMap::get(numDims, 0, {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1), rewriter.getAffineDimExpr(3)}, context);
    AffineMap mapB = AffineMap::get(numDims, 0, {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(3), rewriter.getAffineDimExpr(2)}, context);
    AffineMap mapOut = AffineMap::get(numDims, 0, {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1), rewriter.getAffineDimExpr(2)}, context);

    SmallVector<AffineMap> indexingMaps1 = {mapA, mapB, mapOut};
    SmallVector<utils::IteratorType> iteratorTypes1 = {
        utils::IteratorType::parallel, 
        utils::IteratorType::parallel, 
        utils::IteratorType::parallel,
        utils::IteratorType::reduction};

    // Create the first operation (matmul)
    Value initTensor1 = rewriter.create<tensor::EmptyOp>(
        loc, finalResultType.getShape(), finalResultType.getElementType());
    
    Value cst0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0));
    Value filledTensor = rewriter.create<linalg::FillOp>(loc, cst0, initTensor1).getResult(0);

    auto matmulOp = rewriter.create<linalg::GenericOp>(
        loc,
        TypeRange{finalResultType},
        ValueRange{inputA, inputB},
        ValueRange{filledTensor},
        indexingMaps1,
        iteratorTypes1,
        [&](OpBuilder &builder, Location bodyLoc, ValueRange args) {
            Value mul = builder.create<arith::MulFOp>(bodyLoc, args[0], args[1]);
            Value acc = builder.create<arith::AddFOp>(bodyLoc, args[2], mul);
            builder.create<linalg::YieldOp>(bodyLoc, acc);
        });

    // Create bias expansion for 3D
    auto biasResultType = RankedTensorType::get({1, 1, 32}, rewriter.getF32Type());
    Value biasInput3D = rewriter.create<tensor::ExpandShapeOp>(
        loc, biasResultType, biasInput, SmallVector<ReassociationIndices>{{0, 1}, {2}});

    // Create the second operation (bias + SiLU)
    const int numDims2 = 3;
    AffineMap map3D = AffineMap::get(numDims2, 0, {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1), rewriter.getAffineDimExpr(2)}, context);
    
    SmallVector<AffineMap> indexingMaps2 = {map3D, map3D, map3D};
    SmallVector<utils::IteratorType> iteratorTypes2 = {
        utils::IteratorType::parallel, 
        utils::IteratorType::parallel, 
        utils::IteratorType::parallel};

    Value initTensor2 = rewriter.create<tensor::EmptyOp>(
        loc, finalResultType.getShape(), finalResultType.getElementType());

    auto siluOp = rewriter.create<linalg::GenericOp>(
        loc,
        TypeRange{finalResultType},
        ValueRange{matmulOp.getResult(0), biasInput3D},
        ValueRange{initTensor2},
        indexingMaps2,
        iteratorTypes2,
        [&](OpBuilder &builder, Location bodyLoc, ValueRange args) {
            // Add bias
            Value withBias = builder.create<arith::AddFOp>(bodyLoc, args[0], args[1]);

            // SiLU computation: x * sigmoid(x)
            Value c1 = builder.create<arith::ConstantOp>(bodyLoc, builder.getF32FloatAttr(1.0));
            Value neg = builder.create<arith::NegFOp>(bodyLoc, withBias);
            Value exp = builder.create<math::ExpOp>(bodyLoc, neg);
            Value addOne = builder.create<arith::AddFOp>(bodyLoc, exp, c1);
            Value sig = builder.create<arith::DivFOp>(bodyLoc, c1, addOne);
            Value siluResult = builder.create<arith::MulFOp>(bodyLoc, withBias, sig);

            builder.create<linalg::YieldOp>(bodyLoc, siluResult);
        });

    // Replace the mul operation result with the final fused operation result
    rewriter.replaceAllUsesWith(mulOp.getResult(0), siluOp.getResult(0));

    // Clean up all the original operations
    rewriter.eraseOp(mulOp);
    rewriter.eraseOp(mulInitOp);
    rewriter.eraseOp(sigmoidOp);
    rewriter.eraseOp(sigmoidInitOp);
    rewriter.eraseOp(sigmoidExpandOp);
    rewriter.eraseOp(addOp);
    rewriter.eraseOp(addInitOp);
    rewriter.eraseOp(collapseShapeOp);
    rewriter.eraseOp(batchMatmulOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class LinearSiluFusionPass
    : public PassWrapper<LinearSiluFusionPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinearSiluFusionPass)
  StringRef getArgument() const final { return "linear-silu-fusion"; }
  StringRef getDescription() const final {
    return "Fuse Linear+SiLU operations into a single optimized computation.";
  }
  LinearSiluFusionPass() = default;
  LinearSiluFusionPass(const LinearSiluFusionPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, affine::AffineDialect,
                    math::MathDialect, arith::ArithDialect,
                    memref::MemRefDialect, bufferization::BufferizationDialect>();
  }
};

void LinearSiluFusionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  RewritePatternSet patterns(context);
  patterns.add<LinearSiluFusionPattern>(context);
  GreedyRewriteConfig config;
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns), config))) {
    signalPassFailure();
  }
}

} // end anonymous namespace

namespace mlir {
namespace buddy {
void registerLinearSiluFusionPass() {
  PassRegistration<LinearSiluFusionPass>();
}
} // namespace buddy
} // namespace mlir