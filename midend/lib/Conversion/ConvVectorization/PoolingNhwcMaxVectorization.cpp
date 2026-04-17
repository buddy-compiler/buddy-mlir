//===--------PoolingNhwcMaxVectorization.cpp-------------------------------===//
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

class PoolingNhwcMaxVectorizationPattern : public ConversionPattern {
public:
  explicit PoolingNhwcMaxVectorizationPattern(MLIRContext *context,
                                              int64_t stripParam)
      : ConversionPattern(linalg::PoolingNhwcMaxOp::getOperationName(), 1,
                          context) {
    strip = stripParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Get i1 as the element type for mask vector.
    IntegerType i1 = IntegerType::get(ctx, 1);
    VectorType vectorMaskTy = mlir::VectorType::get({strip}, i1);

    // Get input, kernel and output.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    // Get strides.
    SmallVector<int64_t, 2> strides = {1, 1};
    if (op->hasAttr("strides")) {
      if (auto attr =
              op->getAttrOfType<mlir::DenseIntElementsAttr>("strides")) {
        strides.clear();
        for (auto value : attr.getValues<int64_t>()) {
          strides.push_back(value);
        }
      }
    }
    bool stride1 = strides[0] != 1;
    bool stride2 = strides[1] != 1;
    Value strHeight = arith::ConstantIndexOp::create(rewriter, loc, strides[0]);
    Value strWidth = arith::ConstantIndexOp::create(rewriter, loc, strides[1]);

    // // Get dilations.
    SmallVector<int64_t, 2> dilations = {1, 1};
    if (op->hasAttr("dilations")) {
      if (auto attr =
              op->getAttrOfType<mlir::DenseIntElementsAttr>("dilations")) {
        dilations.clear();
        for (auto value : attr.getValues<int64_t>()) {
          dilations.push_back(value);
        }
      }
    }
    bool dilated1 = dilations[0] != 1;
    bool dilated2 = dilations[1] != 1;
    Value dilHeight =
        arith::ConstantIndexOp::create(rewriter, loc, dilations[0]);
    Value dilWidth = arith::ConstantIndexOp::create(rewriter, loc, dilations[1]);

    // Get ElementType of input.
    Type elementTy =
        mlir::cast<mlir::ShapedType>(input.getType()).getElementType();
    VectorType vectorTy = mlir::VectorType::get({strip}, elementTy);

    // Get Constants.
    const Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    const Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    const Value c2 = arith::ConstantIndexOp::create(rewriter, loc, 2);
    const Value c3 = arith::ConstantIndexOp::create(rewriter, loc, 3);
    const Value vlStep = arith::ConstantIndexOp::create(rewriter, loc, strip);
    const Value zero =
        buddy::insertZeroConstantOp(ctx, rewriter, loc, elementTy);

    // Create pass through vector.
    Value passThroughVec = vector::BroadcastOp::create(rewriter, loc, vectorTy, zero);

    // Get Dimensions of Kernel.
    Value kernelHeight = memref::DimOp::create(rewriter, loc, kernel, c0);
    Value kernelWidth = memref::DimOp::create(rewriter, loc, kernel, c1);

    // Get Dimensions of Outputs.
    Value batch = memref::DimOp::create(rewriter, loc, output, c0);
    Value height = memref::DimOp::create(rewriter, loc, output, c1);
    Value width = memref::DimOp::create(rewriter, loc, output, c2);
    Value channels = memref::DimOp::create(rewriter, loc, output, c3);

    // Calculate the upper bound for vectorized processing
    // - Subtract `vlStep` is to avoid overflow at the vectorization tail.
    // - Add 1 to ensure the final loop runs when the workload length
    //   is divisible by the vector size.
    Value upperBoundTmp = arith::SubIOp::create(rewriter, loc, channels, vlStep);
    Value upperBound = arith::AddIOp::create(rewriter, loc, upperBoundTmp, c1);

    SmallVector<Value, 8> lowerBounds(3, c0);
    SmallVector<Value, 8> uperBounds{batch, height, width};
    SmallVector<int64_t, 8> steps(3, /*Value=*/1);
    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, uperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          // Create strides variables.
          Value tmpIvs1 = ivs[1];
          if (stride1) {
            tmpIvs1 = arith::MulIOp::create(builder, loc, ivs[1], strHeight);
          }
          Value tmpIvs2 = ivs[2];
          if (stride2) {
            tmpIvs2 = arith::MulIOp::create(builder, loc, ivs[2], strWidth);
          }
          // Create strip mining loop.
          auto iterIdx = scf::ForOp::create(builder, 
              loc, c0, upperBound, /*Step=*/vlStep, ValueRange{c0},
              [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
                  ValueRange itrArgs) {
                Value outputVector = vector::LoadOp::create(nestedBuilder, 
                    loc, vectorTy, output,
                    ValueRange{ivs[0], ivs[1], ivs[2], iv});

                auto tmp0 = affine::AffineForOp::create(nestedBuilder, 
                    loc, ValueRange{c0}, builder.getDimIdentityMap(),
                    ValueRange{kernelHeight}, builder.getDimIdentityMap(),
                    /*Step=*/1, ValueRange{outputVector},
                    [&](OpBuilder &builder, Location loc, Value iv0,
                        ValueRange itrArgs0) {
                      // Create dilated[0] variables.
                      Value tmpIvs3 = iv0;
                      if (dilated1) {
                        tmpIvs3 =
                            arith::MulIOp::create(builder, loc, iv0, dilHeight);
                      }
                      Value inputHeight =
                          arith::AddIOp::create(builder, loc, tmpIvs1, tmpIvs3);
                      auto tmp1 = affine::AffineForOp::create(builder, 
                          loc, ValueRange{c0}, builder.getDimIdentityMap(),
                          ValueRange{kernelWidth}, builder.getDimIdentityMap(),
                          /*Step=*/1, ValueRange{itrArgs0[0]},
                          [&](OpBuilder &builder, Location loc, Value iv1,
                              ValueRange itrArgs1) {
                            // Create dilated[1] variables.
                            Value tmpIvs4 = iv1;
                            if (dilated2) {
                              tmpIvs4 = arith::MulIOp::create(builder, loc, iv1,
                                                                      dilWidth);
                            }
                            Value inputWidth = arith::AddIOp::create(builder, 
                                loc, tmpIvs2, tmpIvs4);
                            Value inputVector = vector::LoadOp::create(builder, 
                                loc, vectorTy, input,
                                ValueRange{ivs[0], inputHeight, inputWidth,
                                           iv});
                            // Max
                            Value resultVector;
                            if (auto ty =
                                    llvm::dyn_cast<IntegerType>(elementTy)) {
                              resultVector = arith::MaxSIOp::create(builder, 
                                  loc, inputVector, itrArgs1[0]);
                            } else {
                              resultVector = arith::MaximumFOp::create(builder, 
                                  loc, inputVector, itrArgs1[0]);
                            }
                            affine::AffineYieldOp::create(builder, loc,
                                                                  resultVector);
                          });
                      affine::AffineYieldOp::create(nestedBuilder, 
                          loc, tmp1.getResult(0));
                    });
                vector::StoreOp::create(builder, 
                    loc, tmp0.getResult(0), output,
                    ValueRange{ivs[0], ivs[1], ivs[2], iv});
                Value idx =
                    arith::AddIOp::create(builder, loc, itrArgs[0], vlStep);
                scf::YieldOp::create(builder, loc, idx);
              });
          // Compute the tail size and Process the remaining elements
          // using masked vector operations.
          Value idx = iterIdx.getResult(0);
          Value tailSize = arith::SubIOp::create(builder, loc, channels, idx);
          Value tailCond = arith::CmpIOp::create(rewriter, 
              loc, arith::CmpIPredicate::sgt, tailSize, c0);
          // If the current column does not reach the tail.
          scf::IfOp::create(
              builder, loc, tailCond, [&](OpBuilder &builder, Location loc) {
            // Create mask according to the tail.
            Value tailMask =
                CreateMaskOp::create(builder, loc, vectorMaskTy, tailSize);
            // Masked load output.
            Value maskedOutputVec = MaskedLoadOp::create(builder, 
                loc, vectorTy, output, ValueRange{ivs[0], ivs[1], ivs[2], idx},
                tailMask, passThroughVec);
            auto tmp0 = affine::AffineForOp::create(builder, 
                loc, ValueRange{c0}, builder.getDimIdentityMap(),
                ValueRange{kernelHeight}, builder.getDimIdentityMap(),
                /*Step=*/1, ValueRange{maskedOutputVec},
                [&](OpBuilder &builder, Location loc, Value iv0,
                    ValueRange itrArgs0) {
                  // Create dilated[0] variables.
                  Value tmpIvs3 = iv0;
                  if (dilated1) {
                    tmpIvs3 =
                        arith::MulIOp::create(builder, loc, iv0, dilHeight);
                  }
                  Value inputHeight =
                      arith::AddIOp::create(builder, loc, tmpIvs1, tmpIvs3);
                  auto tmp1 = affine::AffineForOp::create(builder, 
                      loc, ValueRange{c0}, builder.getDimIdentityMap(),
                      ValueRange{kernelWidth}, builder.getDimIdentityMap(),
                      /*Step=*/1, ValueRange{itrArgs0[0]},
                      [&](OpBuilder &builder, Location loc, Value iv1,
                          ValueRange itrArgs1) {
                        // Calculate the index of the input and
                        // output.
                        // Create dilated[1] variables.
                        Value tmpIvs4 = iv1;
                        if (dilated2) {
                          tmpIvs4 =
                              arith::MulIOp::create(builder, loc, iv1, dilWidth);
                        }
                        Value inputWidth =
                            arith::AddIOp::create(builder, loc, iv1, tmpIvs2);
                        // Masked load input and output.
                        Value maskedInputVec = MaskedLoadOp::create(builder, 
                            loc, vectorTy, input,
                            ValueRange{ivs[0], inputHeight, inputWidth, idx},
                            tailMask, passThroughVec);
                        // Max
                        Value resultVec;
                        if (auto ty = llvm::dyn_cast<IntegerType>(elementTy)) {
                          resultVec = arith::MaxSIOp::create(builder, 
                              loc, maskedInputVec, itrArgs1[0]);
                        } else {
                          resultVec = arith::MaximumFOp::create(builder, 
                              loc, maskedInputVec, itrArgs1[0]);
                        }
                        affine::AffineYieldOp::create(builder, loc, resultVec);
                      });
                  affine::AffineYieldOp::create(builder, loc, tmp1.getResult(0));
                });
            // Masked store the result to output.
            MaskedStoreOp::create(builder, 
                loc, output, ValueRange{ivs[0], ivs[1], ivs[2], idx}, tailMask,
                tmp0.getResult(0));
            scf::YieldOp::create(builder, loc);
          });
        });
    // Remove the origin convolution operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t strip;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// PoolingNhwcMaxVectorizationPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling max operations to mixture of
/// Arith + Vector operations.
namespace {
class PoolingNhwcMaxVectorizationPass
    : public PassWrapper<PoolingNhwcMaxVectorizationPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PoolingNhwcMaxVectorizationPass)
  StringRef getArgument() const final {
    return "pooling-nhwc-max-vectorization";
  }
  StringRef getDescription() const final {
    return "Pooling_Nhwc_Max vectorization.";
  }
  PoolingNhwcMaxVectorizationPass() = default;
  PoolingNhwcMaxVectorizationPass(const PoolingNhwcMaxVectorizationPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect, func::FuncDialect>();
  }
  Option<int64_t> strip{*this, "vector-size",
                        llvm::cl::desc("Specify vector type size."),
                        llvm::cl::init(16)};
};
} // end anonymous namespace.

void PoolingNhwcMaxVectorizationPass::runOnOperation() {
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
  patterns.add<PoolingNhwcMaxVectorizationPattern>(context, strip);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerPoolingNhwcMaxVectorizationPass() {
  PassRegistration<PoolingNhwcMaxVectorizationPass>();
}
} // namespace buddy
} // namespace mlir
