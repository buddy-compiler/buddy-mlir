//===--------Conv2dNhwcFhwcVectorization.cpp-------------------------------===//
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
// This file implements the Pooling Nhwc Max Vectorization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/ArrayRef.h"
#include <mlir/Dialect/Affine/Analysis/AffineAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Vector/Transforms/VectorTransforms.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

#include "Utils/Utils.h"

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class Conv2dNhwcFhwcVectorizationPattern : public ConversionPattern {
public:
  explicit Conv2dNhwcFhwcVectorizationPattern(MLIRContext *context,
                                              int64_t vecsizeParam)
      : ConversionPattern(linalg::Conv2DNhwcFhwcOp::getOperationName(), 1,
                          context) {
    vecsize = vecsizeParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Get input, kernel and output.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);

    // Get i1 as the element type for mask vector.
    IntegerType i1 = IntegerType::get(ctx, 1);
    VectorType vectorMaskTy = mlir::VectorType::get({vecsize}, i1);
    // Get ElementType of input and create pass through vector.
    Type elementTy = input.getType().cast<ShapedType>().getElementType();
    VectorType vectorTy = mlir::VectorType::get({vecsize}, elementTy);

    // Get Constants.
    const Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    const Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    const Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    const Value c3 = rewriter.create<arith::ConstantIndexOp>(loc, 3);
    const Value c32 = rewriter.create<arith::ConstantIndexOp>(loc, vecsize);
    const Value zero =
        buddy::insertZeroConstantOp(ctx, rewriter, loc, elementTy);

    // Create pass through vector.
    Value passThroughVec = rewriter.create<SplatOp>(loc, vectorTy, zero);

    // Get Dimensions of Input.
    Value batch = rewriter.create<memref::DimOp>(loc, input, c0);
    Value channels = rewriter.create<memref::DimOp>(loc, input, c3);

    // Get Dimensions of Kernel.
    Value f_o = rewriter.create<memref::DimOp>(loc, kernel, c0);
    Value height_k = rewriter.create<memref::DimOp>(loc, kernel, c1);
    Value width_k = rewriter.create<memref::DimOp>(loc, kernel, c2);

    // Get Dimensions of Outputs.
    Value height_o = rewriter.create<memref::DimOp>(loc, output, c1);
    Value width_o = rewriter.create<memref::DimOp>(loc, output, c2);

    auto inBound = BoolAttr::get(ctx, false);

    AffineExpr d0;
    bindDims(ctx, d0);
    AffineMap vecsizeMap = AffineMap::get(1, 0, {d0.ceilDiv(vecsize)}, ctx);
    // Define AffineMap.
    // The `outputVector` and `inputVector` share the
    // same AffineMap.
    AffineExpr output0, output1, output2, output3;
    bindDims(ctx, output0, output1, output2, output3);
    AffineMap VectorMap = AffineMap::get(
        /*dimCount=*/4, /*symbolCount=*/0, {output2}, ctx);
    SmallVector<Value, 8> lowerBounds(4, c0);
    SmallVector<Value, 8> uperBounds{batch, f_o, channels, height_o};
    SmallVector<int64_t, 8> steps(4, /*Value=*/1);
    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, uperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          // Create vecsize mining loop.
          builder.create<affine::AffineForOp>(
              loc, ValueRange{c0}, builder.getDimIdentityMap(),
              ValueRange{width_o}, vecsizeMap, /*Step=*/1, std::nullopt,
              [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
                  ValueRange itrArgs) {
                // Calculate the tail.
                Value currWidth = builder.create<arith::MulIOp>(loc, iv, c32);
                Value tail =
                    builder.create<arith::SubIOp>(loc, width_o, currWidth);
                Value tailCond = rewriter.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::sge, tail, c32);

                // If the current column does not reach the tail.
                builder.create<scf::IfOp>(
                    loc, tailCond,
                    [&](OpBuilder &builder, Location loc) {
                      // Vectorize the kernel.
                      Value outputVector =
                          nestedBuilder.create<vector::TransferReadOp>(
                              loc, vectorTy, output,
                              ValueRange{ivs[0], ivs[3], currWidth, ivs[1]},
                              VectorMap);

                      auto tmp0 = nestedBuilder.create<affine::AffineForOp>(
                          loc, ValueRange{c0}, builder.getDimIdentityMap(),
                          ValueRange{height_k}, builder.getDimIdentityMap(),
                          /*Step=*/1, ValueRange{outputVector},
                          [&](OpBuilder &builder, Location loc, Value iv0,
                              ValueRange itrArgs0) {
                            auto tmp1 = nestedBuilder.create<
                                affine::AffineForOp>(
                                loc, ValueRange{c0},
                                builder.getDimIdentityMap(),
                                ValueRange{width_k},
                                builder.getDimIdentityMap(),
                                /*Step=*/1, ValueRange{itrArgs0[0]},
                                [&](OpBuilder &builder, Location loc, Value iv1,
                                    ValueRange itrArgs1) {
                                  Value kernelValue =
                                      nestedBuilder.create<memref::LoadOp>(
                                          loc, kernel,
                                          ValueRange{ivs[1], iv0, iv1, ivs[2]});

                                  Value kernelVector =
                                      builder.create<vector::BroadcastOp>(
                                          loc, vectorTy, kernelValue);
                                  Value inputHeight =
                                      nestedBuilder.create<arith::AddIOp>(
                                          loc, ivs[3], iv0);
                                  Value inputWidth =
                                      nestedBuilder.create<arith::AddIOp>(
                                          loc, currWidth, iv1);
                                  Value inputVector =
                                      nestedBuilder
                                          .create<vector::TransferReadOp>(
                                              loc, vectorTy, input,
                                              ValueRange{ivs[0], inputHeight,
                                                         inputWidth, ivs[2]},
                                              VectorMap);
                                  // FMA
                                  Value resultVector;
                                  if (auto ty = llvm::dyn_cast<IntegerType>(
                                          elementTy)) {
                                    Value tmpVector =
                                        nestedBuilder.create<arith::MulIOp>(
                                            loc, inputVector, kernelVector);
                                    resultVector =
                                        nestedBuilder.create<arith::AddIOp>(
                                            loc, tmpVector, itrArgs1[0]);
                                  } else {
                                    resultVector = nestedBuilder.create<FMAOp>(
                                        loc, inputVector, kernelVector,
                                        itrArgs1[0]);
                                  }
                                  nestedBuilder.create<affine::AffineYieldOp>(
                                      loc, resultVector);
                                });
                            nestedBuilder.create<affine::AffineYieldOp>(
                                loc, tmp1.getResult(0));
                          });
                      builder.create<vector::TransferWriteOp>(
                          loc, tmp0.getResult(0), output,
                          ValueRange{ivs[0], ivs[3], currWidth, ivs[1]},
                          VectorMap);
                      builder.create<scf::YieldOp>(loc);
                    },
                    [&](OpBuilder &builder, Location loc) {
                      // Create mask according to the tail.
                      Value tailMask = nestedBuilder.create<CreateMaskOp>(
                          loc, vectorMaskTy, tail);
                      // Masked load output.
                      Value outputVector =
                          nestedBuilder.create<vector::TransferReadOp>(
                              loc, vectorTy, output,
                              ValueRange{ivs[0], ivs[3], currWidth, ivs[1]},
                              AffineMapAttr::get(VectorMap), zero, tailMask,
                              ArrayAttr::get(ctx,
                                             ArrayRef<Attribute>(inBound)));

                      auto tmp0 = nestedBuilder.create<affine::AffineForOp>(
                          loc, ValueRange{c0}, builder.getDimIdentityMap(),
                          ValueRange{height_k}, builder.getDimIdentityMap(),
                          /*Step=*/1, ValueRange{outputVector},
                          [&](OpBuilder &builder, Location loc, Value iv0,
                              ValueRange itrArgs0) {
                            auto tmp1 = nestedBuilder.create<
                                affine::AffineForOp>(
                                loc, ValueRange{c0},
                                builder.getDimIdentityMap(),
                                ValueRange{width_k},
                                builder.getDimIdentityMap(),
                                /*Step=*/1, ValueRange{itrArgs0[0]},
                                [&](OpBuilder &builder, Location loc, Value iv1,
                                    ValueRange itrArgs1) {
                                  Value kernelValue =
                                      nestedBuilder.create<memref::LoadOp>(
                                          loc, kernel,
                                          ValueRange{ivs[1], iv0, iv1, ivs[2]});

                                  Value kernelVector =
                                      builder.create<vector::BroadcastOp>(
                                          loc, vectorTy, kernelValue);
                                  Value inputHeight =
                                      nestedBuilder.create<arith::AddIOp>(
                                          loc, ivs[3], iv0);
                                  Value inputWidth =
                                      nestedBuilder.create<arith::AddIOp>(
                                          loc, currWidth, iv1);
                                  Value inputVector =
                                      nestedBuilder
                                          .create<vector::TransferReadOp>(
                                              loc, vectorTy, input,
                                              ValueRange{ivs[0], inputHeight,
                                                         inputWidth, ivs[2]},
                                              AffineMapAttr::get(VectorMap),
                                              zero, tailMask,
                                              ArrayAttr::get(
                                                  ctx, ArrayRef<Attribute>(
                                                           inBound)));
                                  // FMA
                                  Value resultVector;
                                  if (auto ty = llvm::dyn_cast<IntegerType>(
                                          elementTy)) {
                                    Value tmpVector =
                                        nestedBuilder.create<arith::MulIOp>(
                                            loc, inputVector, kernelVector);
                                    resultVector =
                                        nestedBuilder.create<arith::AddIOp>(
                                            loc, tmpVector, itrArgs1[0]);
                                  } else {
                                    resultVector = nestedBuilder.create<FMAOp>(
                                        loc, inputVector, kernelVector,
                                        itrArgs1[0]);
                                  }
                                  nestedBuilder.create<affine::AffineYieldOp>(
                                      loc, resultVector);
                                });
                            nestedBuilder.create<affine::AffineYieldOp>(
                                loc, tmp1.getResult(0));
                          });
                      builder.create<vector::TransferWriteOp>(
                          loc, tmp0.getResult(0), output,
                          ValueRange{ivs[0], ivs[3], currWidth, ivs[1]},
                          AffineMapAttr::get(VectorMap), tailMask,
                          ArrayAttr::get(ctx, ArrayRef<Attribute>(inBound)));
                      builder.create<scf::YieldOp>(loc);
                    });
                nestedBuilder.create<affine::AffineYieldOp>(nestedLoc);
              });
        });
    // Remove the origin convolution operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t vecsize;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Conv2dNhwcFhwcVectorizationPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling max operations to mixture of
/// Arith + Vector operations.
namespace {
class Conv2dNhwcFhwcVectorizationPass
    : public PassWrapper<Conv2dNhwcFhwcVectorizationPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Conv2dNhwcFhwcVectorizationPass)
  StringRef getArgument() const final {
    return "conv2d-nhwc-fhwc-vectorization";
  }
  StringRef getDescription() const final {
    return "Conv2d_Nhwc_Fhwc Vectorization.";
  }
  Conv2dNhwcFhwcVectorizationPass() = default;
  Conv2dNhwcFhwcVectorizationPass(const Conv2dNhwcFhwcVectorizationPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect, func::FuncDialect>();
  }
  Option<int64_t> vecsize{*this, "vec-size",
                          llvm::cl::desc("Specify vector type size."),
                          llvm::cl::init(8)};
};
} // end anonymous namespace.

void Conv2dNhwcFhwcVectorizationPass::runOnOperation() {
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
  patterns.add<Conv2dNhwcFhwcVectorizationPattern>(context, vecsize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerConv2dNhwcFhwcVectorizationPass() {
  PassRegistration<Conv2dNhwcFhwcVectorizationPass>();
}
} // namespace buddy
} // namespace mlir
