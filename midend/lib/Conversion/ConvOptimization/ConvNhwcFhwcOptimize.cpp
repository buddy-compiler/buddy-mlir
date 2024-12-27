//====- ConvNhwcFhwcOptimize.cpp----------------------------===//
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
// This file implements the Conv2DNhwcFhwcOp optimize.
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class ConvNhwcFhwcOptimizePattern : public ConversionPattern {
public:
  explicit ConvNhwcFhwcOptimizePattern(MLIRContext *context,
                                       int64_t vecSizeParam)
      : ConversionPattern(linalg::Conv2DNhwcFhwcOp::getOperationName(), 1,
                          context) {
    vecSize = vecSizeParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto convOp = dyn_cast_or_null<mlir::linalg::Conv2DNhwcFhwcOp>(op);
    auto loc = op->getLoc();

    // Some constant we need.
    const Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    const Value vecSizeValue =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(vecSize));
    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);
    const AffineExpr s0 = rewriter.getAffineSymbolExpr(0);

    Value input = op->getOperand(0);
    Value filter = op->getOperand(1);
    Value output = op->getOperand(2);

    int strHeight, strWidth, dilHeight, dilWidth;

    // Strides.
    if (!convOp.getStrides()) {
      strHeight = 1;
      strWidth = 1;
    } else {
      strHeight = convOp.getStrides().getValues<int64_t>()[0];
      strWidth = convOp.getStrides().getValues<int64_t>()
                     [convOp.getStrides().getValues<int64_t>().size() - 1];
    }

    // Dilations.
    if (!convOp.getDilations()) {
      dilHeight = 1;
      dilWidth = 1;
    } else {
      dilHeight = convOp.getDilations().getValues<int64_t>()[0];
      dilWidth = convOp.getDilations().getValues<int64_t>()
                     [convOp.getDilations().getValues<int64_t>().size() - 1];
    }

    ShapedType inputTy = input.getType().cast<ShapedType>();
    Type elemTy = inputTy.getElementType();
    VectorType vecTy = VectorType::get(vecSize, elemTy);

    const Value zeroElementType =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemTy));

    // Dims
    Value N = rewriter.create<memref::DimOp>(loc, output, 0);  // N
    Value OH = rewriter.create<memref::DimOp>(loc, output, 1); // OH
    Value OW = rewriter.create<memref::DimOp>(loc, output, 2); // OW
    Value OC = rewriter.create<memref::DimOp>(loc, output, 3); // OC
    Value IC = rewriter.create<memref::DimOp>(loc, input, 3);  // IC
    Value FH = rewriter.create<memref::DimOp>(loc, filter, 1); // FH
    Value FW = rewriter.create<memref::DimOp>(loc, filter, 2); // FW

    // clang format off
    //  Step 1: Create outer most loops.
    // Create the scf::ForallOp operation For N,OH,OW,OC
    rewriter.create<scf::ForallOp>(
        loc, SmallVector<OpFoldResult, 4>({N, OH, OW, OC}), ValueRange{},
        std::nullopt, // No mapping specified in this example
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange loopIndices) {
          Value ivN = loopIndices[0];  // Index for the first dimension N
          Value ivOH = loopIndices[1]; // Index for the second dimension OH
          Value ivOW = loopIndices[2]; // Index for the third dimension OW
          Value ivOC = loopIndices[3]; // Index for the third dimension OC

          Value addRes = nestedBuilder.create<memref::LoadOp>(
              loc, output, ValueRange{ivN, ivOH, ivOW, ivOC});
          // IC
          auto forOp = nestedBuilder.create<scf::ForOp>(
              nestedLoc, c0, IC, vecSizeValue, ValueRange{addRes},
              [&](OpBuilder &builder, Location loc, Value ivIC,
                  ValueRange iargs) {
                Value tVec;
                if (isa<IntegerType>(elemTy)) {
                  tVec = builder.create<vector::BroadcastOp>(loc, vecTy,
                                                             zeroElementType);
                } else {
                  tVec = builder.create<vector::SplatOp>(loc, vecTy,
                                                         zeroElementType);
                }

                Value remainLen = builder.create<affine::AffineMinOp>(
                    loc,
                    AffineMap::get(2, 1, {-d0 + s0, d1}, builder.getContext()),
                    ValueRange{ivIC, vecSizeValue, IC});
                Value remainMask = builder.create<vector::CreateMaskOp>(
                    loc, VectorType::get({vecSize}, rewriter.getI1Type()),
                    ValueRange{remainLen});

                // FH
                auto forOp = builder.create<scf::ForOp>(
                    loc, c0, FH, c1, ValueRange{tVec},
                    [&](OpBuilder &builder, Location loc, Value ivFH,
                        ValueRange iargs) {
                      Value rowInput = builder.create<affine::AffineApplyOp>(
                          loc,
                          AffineMap::get(2, 0, d0 * strHeight + d1 * dilHeight),
                          ValueRange{ivOH, ivFH});
                      Value rowFilter = ivFH;
                      // FW
                      auto forOp = builder.create<scf::ForOp>(
                          loc, c0, FW, c1, ValueRange{iargs[0]},
                          [&](OpBuilder &builder, Location loc, Value ivFW,
                              ValueRange iargs) {
                            Value columnInput =
                                builder.create<affine::AffineApplyOp>(
                                    loc,
                                    AffineMap::get(
                                        2, 0, d0 * strWidth + d1 * dilWidth),
                                    ValueRange{ivOW, ivFW});
                            Value columnFilter = ivFW;
                            Value iVec = builder.create<vector::LoadOp>(
                                loc, vecTy, input,
                                ValueRange{ivN, rowInput, columnInput, ivIC});
                            Value fVec = builder.create<vector::LoadOp>(
                                loc, vecTy, filter,
                                ValueRange{ivOC, rowFilter, columnFilter,
                                           ivIC});
                            Value tVecNext;
                            if (isa<IntegerType>(elemTy)) {
                              Value mulVec = builder.create<arith::MulIOp>(
                                  loc, iVec, fVec);
                              tVecNext = builder.create<arith::AddIOp>(
                                  loc, mulVec, iargs[0]);
                            } else {
                              tVecNext = builder.create<vector::FMAOp>(
                                  loc, vecTy, iVec, fVec, iargs[0]);
                            }

                            builder.create<scf::YieldOp>(loc,
                                                         ValueRange{tVecNext});
                          });
                      builder.create<scf::YieldOp>(
                          loc, ValueRange{forOp.getResult(0)});
                    });
                auto reduceVecOp = builder.create<vector::ReductionOp>(
                    loc, vector::CombiningKind::ADD, forOp.getResult(0));
                auto maskedOp =
                    cast<vector::MaskOp>(mlir::vector::maskOperation(
                        builder, reduceVecOp, remainMask));
                Value reduceVec = maskedOp->getResult(0);
                Value addNext;
                if (isa<IntegerType>(elemTy)) {
                  addNext =
                      builder.create<arith::AddIOp>(loc, iargs[0], reduceVec);
                } else {
                  addNext =
                      builder.create<arith::AddFOp>(loc, iargs[0], reduceVec);
                }
                builder.create<scf::YieldOp>(loc, ValueRange{addNext});
              });

          nestedBuilder.create<memref::StoreOp>(
              loc, forOp.getResult(0), output,
              ValueRange{ivN, ivOH, ivOW, ivOC});
          nestedBuilder.create<scf::InParallelOp>(nestedLoc);
        });
    // clang format on

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t vecSize;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ConvNhwcFhwcOptimizePass
//===----------------------------------------------------------------------===//

namespace {
class ConvNhwcFhwcOptimizePass
    : public PassWrapper<ConvNhwcFhwcOptimizePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvNhwcFhwcOptimizePass)
  StringRef getArgument() const final { return "conv-nhwc-fhwc-optimize"; }
  StringRef getDescription() const final {
    return "Conv2d NHWC FHWC optimize.";
  }
  ConvNhwcFhwcOptimizePass() = default;
  ConvNhwcFhwcOptimizePass(const ConvNhwcFhwcOptimizePass &) {}
  explicit ConvNhwcFhwcOptimizePass(int64_t vecSizeParam) {
    vecSize = vecSizeParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect>();
  }

  Option<int64_t> vecSize{*this, "vec-size", llvm::cl::desc("Vector size."),
                          llvm::cl::init(16)};
};
} // end anonymous namespace.

void ConvNhwcFhwcOptimizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<ConvNhwcFhwcOptimizePattern>(context, vecSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerConvNhwcFhwcOptimizePass() {
  PassRegistration<ConvNhwcFhwcOptimizePass>();
}
} // namespace buddy
} // namespace mlir
