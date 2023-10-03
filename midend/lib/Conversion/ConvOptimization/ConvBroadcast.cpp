//====- ConvBroadcast.cpp --------------------------------------------------===//
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
// This file implements the Conv Broadcast Optmize for linalg.conv_2d_nhwc_hwcf
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/Pass/Pass.h>
#include <iostream>

#include "Utils/Utils.h"

using namespace mlir;
using namespace vector;

//===--------------------------------------------
// Rewrite Pattern
//===--------------------------------------------

namespace {
class ConvBroadcastOptimizePattern : public ConversionPattern {
public:
  explicit ConvBroadcastOptimizePattern(MLIRContext *context, int64_t strideParam,
                                        ArrayRef<int64_t> tileParam)
      : ConversionPattern(linalg::Conv2DNhwcHwcfOp::getOperationName(), 1, context) {
    
    stride = strideParam;
    tileSizes = tileParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();
    // Currently use f32 as the element type.
    FloatType f32 = mlir::FloatType::getF32(ctx);
    // Get i1 as the element type for mask vector.
    IntegerType i1 = IntegerType::get(ctx, 1);
    // Define `*Type`.
    VectorType vectorTy32 = mlir::VectorType::get({stride}, f32);
    VectorType vectorMaskTy = VectorType::get({stride}, i1);
    // Create constant index.
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value c3 = rewriter.create<arith::ConstantIndexOp>(loc, 3);
    Value cStride = rewriter.create<arith::ConstantIndexOp>(loc, stride);
    Value f0 = rewriter.create<arith::ConstantFloatOp>(
      loc, APFloat::getZero(f32.getFloatSemantics()), f32
    );

    // Get input, kernel and output.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    // Create DimOp.
    Value kernelRow = rewriter.create<memref::DimOp>(loc, kernel, c0);
    Value kernelCol = rewriter.create<memref::DimOp>(loc, kernel, c1);
    Value outputRow = rewriter.create<memref::DimOp>(loc, output, c1);
    Value outputCol = rewriter.create<memref::DimOp>(loc, output, c2);
    Value batch = rewriter.create<memref::DimOp>(loc, input, c0);
    Value feature = rewriter.create<memref::DimOp>(loc, kernel, c3);
    Value channel = rewriter.create<memref::DimOp>(loc, kernel, c2);
    // Size of strip mining.
    AffineExpr d0;
    bindDims(ctx, d0);
    AffineMap stripMap = AffineMap::get(1, 0, {d0.ceilDiv(stride)}, ctx);
    SmallVector<Value, 8> lowerBounds(6, c0);
    SmallVector<Value, 8> uperBounds{batch, outputRow, kernelRow, kernelCol, channel, feature};
    SmallVector<int64_t, 8> steps(6, 1);
    affine::buildAffineLoopNest(rewriter, loc, lowerBounds, uperBounds, steps,
    [&](OpBuilder &builder, Location loc, ValueRange ivs) {
      // Create stride loop.
      builder.create<affine::AffineForOp>(
        loc, ValueRange{c0}, builder.getDimIdentityMap(),
        ValueRange{outputCol}, stripMap, 1, std::nullopt,
        [&](OpBuilder &nestedBuilder, Location channelLoc, Value iv, ValueRange itrArgs) {
          // Get element frome kernel.
          Value kernelValue = builder.create<memref::LoadOp>(
            loc, kernel, ValueRange{ivs[2], ivs[3], ivs[4], ivs[5]}
          );
          // Coefficients handle, if kernel item == 0, skip compute
          Value kernelNonZeroCond = buddy::zeroCond(
            builder, loc, f32, kernelValue, buddy::indexToF32(builder, loc, c0)
          );
          builder.create<scf::IfOp>(loc, kernelNonZeroCond,
          [&](OpBuilder &builder, Location loc) {
            // Broadcast element of the kernel.
            Value kernelVector = builder.create<vector::BroadcastOp>(
              loc, vectorTy32, kernelValue
            );
            // Calculate the tail.
            Value currCol = nestedBuilder.create<arith::MulIOp>(loc, iv, cStride);
            Value tail = nestedBuilder.create<arith::SubIOp>(loc, outputCol, currCol);

            Value inputRowTailIdx = builder.create<arith::AddIOp>(loc, ivs[1], ivs[2]);
            //Value outputColTailIdx = builder.create<arith::MulIOp>(loc, iv, cStride);
            Value inputColTailIdx = builder.create<arith::AddIOp>(loc, ivs[3], currCol);
            Value tailMask = builder.create<CreateMaskOp>(loc, vectorMaskTy, tail);
            // Define AffineMap (d0, d1, d2, d3) -> (d2)
            AffineExpr d0, d1, d2, d3;
            bindDims(ctx, d0, d1, d2, d3);
            AffineMap tansposeMap = AffineMap::get(4, 0, {d2}, ctx);
            SmallVector<bool> inBounds(1, true);
            // Load input/output vector from memref.
            Value inputVector = builder.create<vector::TransferReadOp>(
              loc, vectorTy32, input, ValueRange{ivs[0], inputRowTailIdx, inputColTailIdx, ivs[4]},
              AffineMapAttr::get(tansposeMap), f0, tailMask, ArrayAttr::get(ctx, builder.getBoolAttr(true))
            );
            Value outputVector = builder.create<vector::TransferReadOp>(
              loc, vectorTy32, output, ValueRange{ivs[0], ivs[1], currCol, ivs[5]},
              AffineMapAttr::get(tansposeMap), f0, tailMask, ArrayAttr::get(ctx, builder.getBoolAttr(true))
            );
            
            // Multiply input vector and kernel vector then Add OutputVector(FMA).
            Value resultVector = builder.create<FMAOp>(
              loc, inputVector, kernelVector, outputVector
            );
            // Store result vector to output.
            builder.create<vector::TransferWriteOp>(
              loc, resultVector, output, ValueRange{ivs[0], ivs[1], currCol, ivs[5]},
              AffineMapAttr::get(tansposeMap), tailMask, ArrayAttr::get(ctx, builder.getBoolAttr(true))
            );
            builder.create<scf::YieldOp>(loc);
          });
          nestedBuilder.create<affine::AffineYieldOp>(channelLoc);
        }
      );
    });
    // Remove the origin convolution operation
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
  ArrayRef<int64_t> tileSizes;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ConvBroadcastNhwcHwcf
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg conv2d_nhwc_hwcf to mixture of
/// Affine + Vector + Std operations.
namespace
{
class ConvBroadcastNhwcHwcfPass
    : public PassWrapper<ConvBroadcastNhwcHwcfPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvBroadcastNhwcHwcfPass)
  StringRef getArgument() const final { return "conv-broadcast"; }
  StringRef getDescription() const final {
    return "Convolution Broadcast optimize for conv2d_nhwc_hwcf";
  }
  ConvBroadcastNhwcHwcfPass() = default;
  ConvBroadcastNhwcHwcfPass(const ConvBroadcastNhwcHwcfPass &) {}
  explicit ConvBroadcastNhwcHwcfPass(int64_t strideParam,
                                     ArrayRef<int64_t> tileParam) {
    stride = strideParam;
    tile = tileParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &regiistery) const override {
    regiistery.insert<linalg::LinalgDialect, scf::SCFDialect,
                      affine::AffineDialect, VectorDialect, func::FuncDialect>();
  }

  Option<int64_t> stride{*this, "stride",
                         llvm::cl::desc("Stride size."),
                         llvm::cl::init(32)};
  ListOption<int64_t> tile{*this, "tile-sizes", llvm::cl::desc("Tile sizes"),
                           llvm::cl::ZeroOrMore};
};
} // end anonymous namespace

void ConvBroadcastNhwcHwcfPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                          scf::SCFDialect, func::FuncDialect,
                          memref::MemRefDialect, VectorDialect,
                          math::MathDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<ConvBroadcastOptimizePattern>(context, stride, tile);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
  {
      signalPassFailure();
  }
    
}

namespace mlir {
namespace buddy {
void registerConvBroadcastNhwcHwcfPass() {
    PassRegistration<ConvBroadcastNhwcHwcfPass>();
}
}
}
