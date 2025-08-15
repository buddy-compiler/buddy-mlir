//====- TransposeFusionVectorization.cpp
//----------------------------------------===//
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
// This file implements the transpose-matmul-transpose fusion vectorization.
//
//===----------------------------------------------------------------------===//
#include <iostream>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Dialect/Vector/Transforms/VectorTransforms.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewriter Pattern
//===----------------------------------------------------------------------===//

namespace {

// TransposeFusion vectorization pattern
class TransposeFusionVectorizationPattern : public ConversionPattern {
public:
  explicit TransposeFusionVectorizationPattern(MLIRContext *context,
                                               int64_t vecSizeParam)
      : ConversionPattern(tosa::MatMulOp::getOperationName(), 1, context) {
    vecSize = vecSizeParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOpResult(0);

    auto reshapeBOp = B.getDefiningOp<tosa::ReshapeOp>();
    if (!reshapeBOp) {
      return failure();
    }
    auto transposeBOp =
        reshapeBOp.getOperand().getDefiningOp<tosa::TransposeOp>();
    if (!transposeBOp) {
      return failure();
    }
    auto reshapeCUserIt = C.getUsers().begin();
    if (reshapeCUserIt == C.getUsers().end()) {
      return failure();
    }
    Operation *reshapeCOp = *reshapeCUserIt;
    if (!isa<tosa::ReshapeOp>(reshapeCOp)) {
      return failure();
    }

    auto transposeCUserIt = reshapeCOp->getOpResult(0).getUsers().begin();
    if (transposeCUserIt == reshapeCOp->getOpResult(0).getUsers().end()) {
      return failure();
    }
    Operation *transposeCOp = *transposeCUserIt;
    if (!isa<tosa::TransposeOp>(transposeCOp)) {
      return failure();
    }

    auto nextUserIt = transposeCOp->getOpResult(0).getUsers().begin();
    if (nextUserIt == transposeCOp->getOpResult(0).getUsers().end()) {
      return failure();
    }
    Operation *nextUserOp = *nextUserIt;

    // Get i1 as the element type for mask vector.
    IntegerType i1 = IntegerType::get(ctx, 1);
    VectorType vectorMaskTy = mlir::VectorType::get({vecSize}, i1);
    // Acquire the element type of input tensors.
    ShapedType AType = A.getType().cast<ShapedType>();
    Type elementType = AType.getElementType();
    VectorType vectorTy = mlir::VectorType::get({vecSize}, elementType);

    ShapedType newBType =
        transposeBOp.getOperand(0).getType().cast<ShapedType>();
    ShapedType newCType =
        transposeCOp->getOpResult(0).getType().cast<ShapedType>();

    // Define constants.
    const Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    const Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    const Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    const Value c3 = rewriter.create<arith::ConstantIndexOp>(loc, 3);
    const Value vlStep = rewriter.create<arith::ConstantIndexOp>(loc, vecSize);
    const Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));

    // Create pass through vector.
    Value passThroughVec = rewriter.create<SplatOp>(loc, vectorTy, zero);
    Value newA = rewriter.create<bufferization::ToMemrefOp>(
        loc, MemRefType::get(AType.getShape(), elementType), A);
    Value newB = rewriter.create<bufferization::ToMemrefOp>(
        loc, MemRefType::get(newBType.getShape(), elementType),
        transposeBOp.getOperand(0));
    Value newC = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(newCType.getShape(), elementType));

    // Get dimensions of input tensors.
    Value batch = rewriter.create<memref::DimOp>(loc, newA, c0);
    Value aRow = rewriter.create<memref::DimOp>(loc, newA, c1);
    Value aCol = rewriter.create<memref::DimOp>(loc, newA, c2);
    Value bCol = rewriter.create<memref::DimOp>(loc, newB, c3);

    // Calculate the upper bound for vectorized processing
    // - Subtract `vlStep` is to avoid overflow at the vectorization tail.
    // - Add 1 to ensure the final loop runs when the workload length
    //   is divisible by the vector size.
    Value upperBoundTmp = rewriter.create<arith::SubIOp>(loc, bCol, vlStep);
    Value upperBound = rewriter.create<arith::AddIOp>(loc, upperBoundTmp, c1);

    affine::buildAffineLoopNest(
        rewriter, loc, {c0, c0}, {batch, aRow}, /*Step=*/{1, 1},
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          auto iterIdx = builder.create<scf::ForOp>(
              loc, c0, upperBound, /*Step=*/vlStep, ValueRange{c0},
              [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
                  ValueRange itrArgs) {
                auto iterVec = nestedBuilder.create<scf::ForOp>(
                    nestedLoc, c0, aCol, /*Step=*/c1, ValueRange{passThroughVec},
                    [&](OpBuilder &nestedBuilder0, Location nestedLoc0,
                        Value iv0, ValueRange itrArgs0) {
                      Value aVal = nestedBuilder0.create<memref::LoadOp>(
                          nestedLoc0, elementType, newA,
                          ValueRange{ivs[0], ivs[1], iv0});
                      Value aVec = nestedBuilder0.create<vector::SplatOp>(
                          nestedLoc0, vectorTy, aVal);
                      Value bVec = nestedBuilder0.create<vector::LoadOp>(
                          nestedLoc0, vectorTy, newB,
                          ValueRange{c0, iv0, ivs[0], iv});
                      // Compute the result vector either through integer
                      // multiplication and addition or fused multiply-add
                      // based on the element type.
                      Value tmpVec;
                      if (isa<IntegerType>(elementType)) {
                        Value mulVec = nestedBuilder0.create<arith::MulIOp>(
                            nestedLoc0, aVec, bVec);
                        tmpVec = nestedBuilder0.create<arith::AddIOp>(
                            nestedLoc0, mulVec, itrArgs0[0]);
                      } else {
                        tmpVec = nestedBuilder0.create<vector::FMAOp>(
                            nestedLoc0, vectorTy, aVec, bVec, itrArgs0[0]);
                      }
                      builder.create<scf::YieldOp>(loc, tmpVec);
                    });
                nestedBuilder.create<vector::StoreOp>(
                    nestedLoc, iterVec.getResult(0), newC,
                    ValueRange{c0, ivs[1], ivs[0], iv});
                Value idx =
                    nestedBuilder.create<arith::AddIOp>(nestedLoc, iv, vlStep);
                nestedBuilder.create<scf::YieldOp>(nestedLoc, idx);
              });
          // Compute the tail size and Process the remaining elements
          // using masked vector operations.
          Value idx = iterIdx.getResult(0);
          Value tailSize = builder.create<arith::SubIOp>(loc, bCol, idx);
          Value tailMask =
              builder.create<CreateMaskOp>(loc, vectorMaskTy, tailSize);
          auto iterVec = builder.create<scf::ForOp>(
              loc, c0, aCol, /*Step=*/c1, ValueRange{passThroughVec},
              [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
                  ValueRange itrArgs) {
                Value aVal = nestedBuilder.create<memref::LoadOp>(
                    nestedLoc, elementType, newA,
                    ValueRange{ivs[0], ivs[1], iv});
                Value aVec = nestedBuilder.create<vector::SplatOp>(
                    nestedLoc, vectorTy, aVal);
                Value bVec = nestedBuilder.create<MaskedLoadOp>(
                    nestedLoc, vectorTy, newB,
                    ValueRange{c0, iv, ivs[0], idx}, tailMask,
                    passThroughVec);

                // Compute the result vector either through integer
                // multiplication and addition or fused multiply-add
                // based on the element type.
                Value tmpVec;
                if (isa<IntegerType>(elementType)) {
                  Value mulVec = nestedBuilder.create<arith::MulIOp>(
                      nestedLoc, aVec, bVec);
                  tmpVec = nestedBuilder.create<arith::AddIOp>(
                      nestedLoc, mulVec, itrArgs[0]);
                } else {
                  tmpVec = nestedBuilder.create<vector::FMAOp>(
                      nestedLoc, vectorTy, aVec, bVec, itrArgs[0]);
                }
                builder.create<scf::YieldOp>(nestedLoc, tmpVec);
              });
          builder.create<MaskedStoreOp>(loc, newC,
                                        ValueRange{c0, ivs[1], ivs[0], idx},
                                        tailMask, iterVec.getResult(0));
        });
    Value output = rewriter.create<bufferization::ToTensorOp>(
        loc, newCType, newC, /*restrict=*/true);

    rewriter.eraseOp(reshapeBOp);
    rewriter.eraseOp(transposeBOp);
    rewriter.eraseOp(op);
    rewriter.eraseOp(reshapeCOp);
    rewriter.replaceOp(transposeCOp, output);
    return success();
  }

private:
  int64_t vecSize;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// TransposeFusionVectorizationPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector operations.
namespace {
class TransposeFusionVectorizationPass
    : public PassWrapper<TransposeFusionVectorizationPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransposeFusionVectorizationPass)
  StringRef getArgument() const final {
    return "transpose-fusion-vectorization";
  }
  StringRef getDescription() const final {
    return "Transpose Fusion Vectorization.";
  }
  TransposeFusionVectorizationPass() = default;
  TransposeFusionVectorizationPass(const TransposeFusionVectorizationPass &) {}
  explicit TransposeFusionVectorizationPass(int64_t vecSizeParam) {
    vecSize = vecSizeParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    bufferization::BufferizationDialect, affine::AffineDialect,
                    tosa::TosaDialect, VectorDialect>();
  }

  Option<int64_t> vecSize{*this, "vector-size",
                          llvm::cl::desc("Affine Vector size."),
                          llvm::cl::init(32)};
};
} // end anonymous namespace.

void TransposeFusionVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                         scf::SCFDialect, memref::MemRefDialect,
                         linalg::LinalgDialect, VectorDialect,
                         bufferization::BufferizationDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<TransposeFusionVectorizationPattern>(context, vecSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerTransposeFusionVectorizationPass() {
  PassRegistration<TransposeFusionVectorizationPass>();
}
} // namespace buddy
} // namespace mlir
