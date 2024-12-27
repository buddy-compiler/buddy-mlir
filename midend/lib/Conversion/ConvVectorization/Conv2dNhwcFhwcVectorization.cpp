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
    // Get Strides.
    SmallVector<int64_t, 2> strides = {1, 1};
    if (op->hasAttr("strides")) {
      strides.clear();
      for (auto value : op->getAttrOfType<mlir::DenseIntElementsAttr>("strides")
                            .getValues<int64_t>()) {
        strides.push_back(value);
      }
    }
    bool stride1 = strides[0] != 1;
    bool stride2 = strides[1] != 1;
    Value strHeight = rewriter.create<arith::ConstantIndexOp>(loc, strides[0]);
    Value strWidth = rewriter.create<arith::ConstantIndexOp>(loc, strides[1]);

    // Get Dilations.
    SmallVector<int64_t, 2> dilations = {1, 1};
    if (op->hasAttr("dilations")) {
      dilations.clear();
      for (auto value :
           op->getAttrOfType<mlir::DenseIntElementsAttr>("dilations")
               .getValues<int64_t>()) {
        dilations.push_back(value);
      }
    }
    bool dilated1 = dilations[0] != 1;
    bool dilated2 = dilations[1] != 1;
    Value dilHeight =
        rewriter.create<arith::ConstantIndexOp>(loc, dilations[0]);
    Value dilWidth = rewriter.create<arith::ConstantIndexOp>(loc, dilations[1]);

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
    const Value vlStep = rewriter.create<arith::ConstantIndexOp>(loc, vecsize);
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

    // Calculate the upper bound for vectorized processing
    // - Subtract `vlStep` is to avoid overflow at the vectorization tail.
    // - Add 1 to ensure the final loop runs when the workload length
    //   is divisible by the vector size.
    Value upperBoundTmp = rewriter.create<arith::SubIOp>(loc, channels, vlStep);
    Value upperBound = rewriter.create<arith::AddIOp>(loc, upperBoundTmp, c1);

    SmallVector<Value, 8> lowerBounds(4, c0);
    SmallVector<Value, 8> uperBounds{batch, height_o, width_o, f_o};
    SmallVector<int64_t, 8> steps(4, /*Value=*/1);
    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, uperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          // Create strides variables.
          Value tmpIvs1 = ivs[1];
          if (stride1) {
            tmpIvs1 = builder.create<arith::MulIOp>(loc, ivs[1], strHeight);
          }
          Value tmpIvs2 = ivs[2];
          if (stride2) {
            tmpIvs2 = builder.create<arith::MulIOp>(loc, ivs[2], strWidth);
          }
          Value tmp_result = builder.create<memref::LoadOp>(
              loc, elementTy, output,
              ValueRange{ivs[0], ivs[1], ivs[2], ivs[3]});
          // Create vecsize mining loop.
          auto iter_val = builder.create<scf::ForOp>(
              loc, c0, upperBound, /*Step=*/vlStep, ValueRange{c0, tmp_result},
              [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
                  ValueRange itrArgs) {
                auto tmp0 = nestedBuilder.create<affine::AffineForOp>(
                    nestedLoc, ValueRange{c0}, builder.getDimIdentityMap(),
                    ValueRange{height_k}, builder.getDimIdentityMap(),
                    /*Step=*/1, ValueRange{itrArgs[1]},
                    [&](OpBuilder &builder, Location loc, Value iv0,
                        ValueRange itrArgs0) {
                      // Create dilated[0] variables.
                      Value tmpIvs3 = iv0;
                      if (dilated1) {
                        tmpIvs3 =
                            builder.create<arith::MulIOp>(loc, iv0, dilHeight);
                      }
                      Value inputHeight =
                          builder.create<arith::AddIOp>(loc, tmpIvs1, tmpIvs3);
                      auto tmp1 = builder.create<affine::AffineForOp>(
                          loc, ValueRange{c0}, builder.getDimIdentityMap(),
                          ValueRange{width_k}, builder.getDimIdentityMap(),
                          /*Step=*/1, ValueRange{itrArgs0[0]},
                          [&](OpBuilder &builder, Location loc, Value iv1,
                              ValueRange itrArgs1) {
                            // Create dilated[1] variables.
                            Value tmpIvs4 = iv1;
                            if (dilated2) {
                              tmpIvs4 = builder.create<arith::MulIOp>(loc, iv1,
                                                                      dilWidth);
                            }
                            Value inputWidth = builder.create<arith::AddIOp>(
                                loc, tmpIvs2, tmpIvs4);
                            Value inputVector = builder.create<vector::LoadOp>(
                                loc, vectorTy, input,
                                ValueRange{ivs[0], inputHeight, inputWidth,
                                           iv});
                            Value kernelVector = builder.create<vector::LoadOp>(
                                loc, vectorTy, kernel,
                                ValueRange{ivs[3], iv0, iv1, iv});
                            // FMA
                            Value resultVal;
                            if (auto ty =
                                    llvm::dyn_cast<IntegerType>(elementTy)) {
                              Value tmpVec = builder.create<arith::MulIOp>(
                                  loc, inputVector, kernelVector);
                              Value tmpVal =
                                  builder.create<vector::ReductionOp>(
                                      loc, vector::CombiningKind::ADD, tmpVec,
                                      ::mlir::arith::FastMathFlags::none);
                              resultVal = builder.create<arith::AddIOp>(
                                  loc, tmpVal, itrArgs1[0]);
                            } else {
                              Value tmpVec = builder.create<arith::MulFOp>(
                                  loc, inputVector, kernelVector);
                              Value tmpVal =
                                  builder.create<vector::ReductionOp>(
                                      loc, vector::CombiningKind::ADD, tmpVec,
                                      ::mlir::arith::FastMathFlags::none);
                              resultVal = builder.create<arith::AddFOp>(
                                  loc, tmpVal, itrArgs1[0]);
                            }
                            builder.create<affine::AffineYieldOp>(loc,
                                                                  resultVal);
                          });
                      nestedBuilder.create<affine::AffineYieldOp>(
                          nestedLoc, tmp1.getResult(0));
                    });
                Value idx =
                    builder.create<arith::AddIOp>(loc, itrArgs[0], vlStep);
                builder.create<scf::YieldOp>(
                    loc, ValueRange{idx, tmp0.getResult(0)});
              });
          // Compute the tail size and Process the remaining elements
          // using masked vector operations.

          Value idx = iter_val.getResult(0);
          Value tailSize = builder.create<arith::SubIOp>(loc, channels, idx);
          Value tailCond = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sgt, tailSize, c0);
          // If the current column does not reach the tail.
          builder.create<scf::IfOp>(
              loc, tailCond,
              [&](OpBuilder &builder, Location loc) {
                Value tailMask =
                    builder.create<CreateMaskOp>(loc, vectorMaskTy, tailSize);
                auto tmp0 = builder.create<affine::AffineForOp>(
                    loc, ValueRange{c0}, builder.getDimIdentityMap(),
                    ValueRange{height_k}, builder.getDimIdentityMap(),
                    /*Step=*/1, ValueRange{iter_val.getResult(1)},
                    [&](OpBuilder &builder, Location loc, Value iv0,
                        ValueRange itrArgs0) {
                      // Create dilated[0] variables.
                      Value tmpIvs3 = iv0;
                      if (dilated1) {
                        tmpIvs3 =
                            builder.create<arith::MulIOp>(loc, iv0, dilHeight);
                      }
                      Value inputHeight =
                          builder.create<arith::AddIOp>(loc, tmpIvs1, tmpIvs3);
                      auto tmp1 = builder.create<affine::AffineForOp>(
                          loc, ValueRange{c0}, builder.getDimIdentityMap(),
                          ValueRange{width_k}, builder.getDimIdentityMap(),
                          /*Step=*/1, ValueRange{itrArgs0[0]},
                          [&](OpBuilder &builder, Location loc, Value iv1,
                              ValueRange itrArgs1) {
                            // Create dilated[1] variables.
                            Value tmpIvs4 = iv1;
                            if (dilated2) {
                              tmpIvs4 = builder.create<arith::MulIOp>(loc, iv1,
                                                                      dilWidth);
                            }
                            Value inputWidth = builder.create<arith::AddIOp>(
                                loc, tmpIvs2, tmpIvs4);
                            Value inputVec = builder.create<MaskedLoadOp>(
                                loc, vectorTy, input,
                                ValueRange{ivs[0], inputHeight, inputWidth,
                                           idx},
                                tailMask, passThroughVec);
                            Value kernelVec = builder.create<MaskedLoadOp>(
                                loc, vectorTy, kernel,
                                ValueRange{ivs[3], iv0, iv1, idx}, tailMask,
                                passThroughVec);
                            // FMA
                            Value resultVal;
                            if (auto ty =
                                    llvm::dyn_cast<IntegerType>(elementTy)) {
                              Value tmpVec = builder.create<arith::MulIOp>(
                                  loc, inputVec, kernelVec);
                              Value tmpVal =
                                  builder.create<vector::ReductionOp>(
                                      loc, vector::CombiningKind::ADD, tmpVec,
                                      ::mlir::arith::FastMathFlags::none);
                              resultVal = builder.create<arith::AddIOp>(
                                  loc, tmpVal, itrArgs1[0]);
                            } else {
                              Value tmpVec = builder.create<arith::MulFOp>(
                                  loc, inputVec, kernelVec);
                              Value tmpVal =
                                  builder.create<vector::ReductionOp>(
                                      loc, vector::CombiningKind::ADD, tmpVec,
                                      ::mlir::arith::FastMathFlags::none);
                              resultVal = builder.create<arith::AddFOp>(
                                  loc, tmpVal, itrArgs1[0]);
                            }
                            builder.create<affine::AffineYieldOp>(loc,
                                                                  resultVal);
                          });
                      builder.create<affine::AffineYieldOp>(loc,
                                                            tmp1.getResult(0));
                    });
                builder.create<memref::StoreOp>(
                    loc, tmp0.getResult(0), output,
                    ValueRange{ivs[0], ivs[1], ivs[2], ivs[3]});
                builder.create<scf::YieldOp>(loc);
              },
              [&](OpBuilder &builder, Location loc) {
                builder.create<memref::StoreOp>(
                    loc, iter_val.getResult(1), output,
                    ValueRange{ivs[0], ivs[1], ivs[2], ivs[3]});
                builder.create<scf::YieldOp>(loc);
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
