//====- CBPoolingVectorization.cpp ----------------------------------------===//
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
// This file implements the pooling vectorization.
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
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

// PoolingNhwcSum vectorization pattern
class CBPoolingNhwcSumVectorizationPattern : public ConversionPattern {
public:
  explicit CBPoolingNhwcSumVectorizationPattern(MLIRContext *context,
                                                int64_t stripParam)
      : ConversionPattern(linalg::PoolingNhwcSumOp::getOperationName(), 1,
                          context) {
    strip = stripParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    // Get input, kernel and output.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    // Element type.
    MemRefType inputMemRefTy = input.getType().dyn_cast<MemRefType>();
    // Element type.
    FloatType fTy = inputMemRefTy.getElementType().dyn_cast<FloatType>();
    // Constants.
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value c3 = rewriter.create<arith::ConstantIndexOp>(loc, 3);
    // Dimensions of the input.
    Value batch = rewriter.create<memref::DimOp>(loc, input, c0);
    Value height = rewriter.create<memref::DimOp>(loc, input, c1);
    Value width = rewriter.create<memref::DimOp>(loc, input, c2);
    Value channels = rewriter.create<memref::DimOp>(loc, input, c3);
    // Strides.
    auto strides = op->getAttrOfType<mlir::DenseIntElementsAttr>("strides")
                       .getValues<int64_t>();
    Value strHeight = rewriter.create<arith::ConstantIndexOp>(loc, strides[0]);
    Value strWidth = rewriter.create<arith::ConstantIndexOp>(loc, strides[1]);
    // Dilations.
    auto dilations = op->getAttrOfType<mlir::DenseIntElementsAttr>("dilations")
                         .getValues<int64_t>();
    bool dilated = dilations[0] != 1 || dilations[1] != 1;
    // Kernel shape.
    MemRefType kernelTy = kernel.getType().dyn_cast<MemRefType>();
    SmallVector<int64_t> kernelShape;
    bool dynamicKernel = false;
    for (unsigned i = 0; i < kernelTy.getRank(); i++) {
      if (!kernelTy.isDynamicDim(i)) {
        kernelShape.push_back(kernelTy.getDimSize(i));
      } else {
        dynamicKernel = true;
        break;
      }
    }
    if (dynamicKernel) {
      return failure();
    }
    // Pooling window shape.
    Value winHeight = rewriter.create<arith::ConstantIndexOp>(
        loc, dilations[0] * (kernelShape[0] - 1) + 1);
    Value winWidth = rewriter.create<arith::ConstantIndexOp>(
        loc, dilations[1] * (kernelShape[1] - 1) + 1);
    // Output shape.
    Value w0 = rewriter.create<arith::SubIOp>(loc, height, winHeight);
    Value w1 = rewriter.create<arith::SubIOp>(loc, width, winWidth);
    Value ubHeight = rewriter.create<arith::AddIOp>(loc, w0, strHeight);
    Value ubWidth = rewriter.create<arith::AddIOp>(loc, w1, strWidth);
    // Kernel width.
    Value kHeight =
        rewriter.create<arith::ConstantIndexOp>(loc, kernelShape[0]);
    Value kWidth = rewriter.create<arith::ConstantIndexOp>(loc, kernelShape[1]);
    // Kernel size.
    int64_t kSize = kernelShape[0] * kernelShape[1];
    // Vector type.
    VectorType vecTy = VectorType::get({kSize}, fTy);
    // Loop arguments.
    SmallVector<Value, 4> lowerBounds(4, c0);
    SmallVector<Value, 4> upperBounds{batch, ubHeight, ubWidth, channels};
    SmallVector<Value, 4> steps{c1, strHeight, strWidth, c1};
    // Allocate pooling window.
    Value window =
        rewriter.create<memref::AllocOp>(loc, MemRefType::get({kSize}, fTy));
    if (dilated) {
      mlir::scf::buildLoopNest(
          rewriter, loc, lowerBounds, upperBounds, steps, {},
          [&](OpBuilder &builder, Location loc, ValueRange ivs,
              ValueRange args) -> scf::ValueVector {
            // Dialtions.
            Value dilHeight =
                rewriter.create<arith::ConstantIndexOp>(loc, dilations[0]);
            Value dilWidth =
                rewriter.create<arith::ConstantIndexOp>(loc, dilations[1]);
            SmallVector<Value, 2> nestedLowerBounds{ivs[1], ivs[2]};
            Value uHeight =
                rewriter.create<arith::AddIOp>(loc, ivs[1], winHeight);
            Value uWidth =
                rewriter.create<arith::AddIOp>(loc, ivs[2], winWidth);
            SmallVector<Value, 2> nestedUpperBounds{uHeight, uWidth};
            SmallVector<Value, 2> nestedSteps{dilHeight, dilWidth};
            mlir::scf::buildLoopNest(
                rewriter, loc, nestedLowerBounds, nestedUpperBounds,
                nestedSteps, {},
                [&](OpBuilder &nestedBuilder, Location nestedLoc,
                    ValueRange nestedIvs,
                    ValueRange nestedArgs) -> scf::ValueVector {
                  // Load value from pooling window.
                  Value val = builder.create<memref::LoadOp>(
                      loc, input,
                      ValueRange{ivs[0], nestedIvs[0], nestedIvs[1], ivs[3]});
                  // Compute index.
                  Value i0 =
                      builder.create<arith::SubIOp>(loc, nestedIvs[0], ivs[1]);
                  Value i = builder.create<arith::DivUIOp>(loc, i0, dilHeight);
                  Value j0 =
                      builder.create<arith::SubIOp>(loc, nestedIvs[1], ivs[2]);
                  Value j = builder.create<arith::DivUIOp>(loc, j0, dilWidth);
                  Value index0 = builder.create<arith::MulIOp>(loc, i, kWidth);
                  Value index = builder.create<arith::AddIOp>(loc, index0, j);
                  // Store value into memref.
                  builder.create<memref::StoreOp>(loc, val, window, index);
                  return {};
                });
            // Load into a vector.
            Value vec = rewriter.create<vector::TransferReadOp>(
                loc, vecTy, window, ValueRange{c0});
            // Reduce vector.
            Value res = rewriter.create<vector::ReductionOp>(
                loc, vector::CombiningKind::ADD, vec);
            // Output indices.
            Value outHeight =
                rewriter.create<arith::DivUIOp>(loc, ivs[1], strHeight);
            Value outWidth =
                rewriter.create<arith::DivUIOp>(loc, ivs[2], strWidth);
            // Store value into output.
            rewriter.create<memref::StoreOp>(
                loc, res, output,
                ValueRange{ivs[0], outHeight, outWidth, ivs[3]});
            return {};
          });
    } else {
      mlir::scf::buildLoopNest(
          rewriter, loc, lowerBounds, upperBounds, steps, {},
          [&](OpBuilder &builder, Location loc, ValueRange ivs,
              ValueRange args) -> scf::ValueVector {
            SmallVector<Value, 2> nestedLowerBounds{ivs[1], ivs[2]};
            Value uHeight =
                rewriter.create<arith::AddIOp>(loc, ivs[1], kHeight);
            Value uWidth = rewriter.create<arith::AddIOp>(loc, ivs[2], kWidth);
            SmallVector<Value, 2> nestedUpperBounds{uHeight, uWidth};
            SmallVector<Value, 2> nestedSteps{c1, c1};
            mlir::scf::buildLoopNest(
                rewriter, loc, nestedLowerBounds, nestedUpperBounds,
                nestedSteps, {},
                [&](OpBuilder &nestedBuilder, Location nestedLoc,
                    ValueRange nestedIvs,
                    ValueRange nestedArgs) -> scf::ValueVector {
                  // Load value from pooling window.
                  Value val = builder.create<memref::LoadOp>(
                      loc, input,
                      ValueRange{ivs[0], nestedIvs[0], nestedIvs[1], ivs[3]});
                  // Compute index.
                  Value i =
                      builder.create<arith::SubIOp>(loc, nestedIvs[0], ivs[1]);
                  Value j =
                      builder.create<arith::SubIOp>(loc, nestedIvs[1], ivs[2]);
                  Value index0 = builder.create<arith::MulIOp>(loc, i, kWidth);
                  Value index = builder.create<arith::AddIOp>(loc, index0, j);
                  // Store value into memref.
                  builder.create<memref::StoreOp>(loc, val, window, index);
                  return {};
                });
            // Load into a vector.
            Value vec = rewriter.create<vector::TransferReadOp>(
                loc, vecTy, window, ValueRange{c0});
            // Reduce vector.
            Value res = rewriter.create<vector::ReductionOp>(
                loc, vector::CombiningKind::ADD, vec);
            // Output indices.
            Value outHeight =
                rewriter.create<arith::DivUIOp>(loc, ivs[1], strHeight);
            Value outWidth =
                rewriter.create<arith::DivUIOp>(loc, ivs[2], strWidth);
            // Store value into output.
            rewriter.create<memref::StoreOp>(
                loc, res, output,
                ValueRange{ivs[0], outHeight, outWidth, ivs[3]});
            return {};
          });
    }
    // Deallocate pooling window.
    rewriter.create<memref::DeallocOp>(loc, window);
    // Remove the origin pooling operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t strip;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// PoolingVectorizationPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector operations.
namespace {
class PoolingVectorizationPass
    : public PassWrapper<PoolingVectorizationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PoolingVectorizationPass)
  StringRef getArgument() const final { return "pooling-vectorization"; }
  StringRef getDescription() const final { return "Pooling vectorization."; }
  PoolingVectorizationPass() = default;
  PoolingVectorizationPass(const PoolingVectorizationPass &) {}
  explicit PoolingVectorizationPass(int64_t stripParam) { strip = stripParam; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect, AffineDialect,
                    VectorDialect>();
  }

  Option<int64_t> strip{*this, "strip-mining",
                        llvm::cl::desc("Strip mining size."),
                        llvm::cl::init(32)};
};
} // end anonymous namespace.

void PoolingVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, AffineDialect, scf::SCFDialect,
                       memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<CBPoolingNhwcSumVectorizationPattern>(context, strip);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerPoolingVectorizationPass() {
  PassRegistration<PoolingVectorizationPass>();
}
} // namespace buddy
} // namespace mlir
