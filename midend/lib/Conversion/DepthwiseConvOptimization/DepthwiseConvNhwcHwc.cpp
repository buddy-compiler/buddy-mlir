//====- DepthwiseConvNhwcHwc.cpp
//--------------------------------------------------===//
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
// This file implements the DepthwiseConvNhwcHwc optimize.
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

class DepthwiseConv2DNhwcHwcOptimizePattern : public ConversionPattern {
public:
  explicit DepthwiseConv2DNhwcHwcOptimizePattern(MLIRContext *context,
                                                 int64_t vecSizeParam)
      : ConversionPattern(linalg::DepthwiseConv2DNhwcHwcOp::getOperationName(),
                          1, context) {
    vecSize = vecSizeParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto convOp = dyn_cast_or_null<mlir::linalg::DepthwiseConv2DNhwcHwcOp>(op);
    auto loc = op->getLoc();

    // Some constant we need.
    const Value c0 =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
    const Value c1 =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1));

    const Value vecSizeValue =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(vecSize));
    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);
    // TODO: remove s0?
    // const AffineExpr s0 = rewriter.getAffineSymbolExpr(0);

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

    ShapedType inputTy = mlir::cast<mlir::ShapedType>(input.getType());
    Type elemTy = inputTy.getElementType();
    VectorType vecTy = VectorType::get(vecSize, elemTy);

    const Value zeroElementType =
        arith::ConstantOp::create(rewriter, loc, rewriter.getZeroAttr(elemTy));

    Value zeroElementTypeVec;
    if (isa<IntegerType>(elemTy)) {
      zeroElementTypeVec =
          vector::BroadcastOp::create(rewriter, loc, vecTy, zeroElementType);
    } else {
      zeroElementTypeVec =
          vector::BroadcastOp::create(rewriter, loc, vecTy, zeroElementType);
    }
    // Dims
    Value N = memref::DimOp::create(rewriter, loc, output, 0);  // N
    Value OH = memref::DimOp::create(rewriter, loc, output, 1); // OH
    Value OW = memref::DimOp::create(rewriter, loc, output, 2); // OW
    Value OC = memref::DimOp::create(rewriter, loc, output, 3); // OC/FC/IC

    Value applyOC = affine::AffineApplyOp::create(rewriter, 
        loc, AffineMap::get(1, 0, d0.floorDiv(vecSize) * vecSize), OC);
    Value tailLength = affine::AffineApplyOp::create(rewriter, 
        loc, AffineMap::get(1, 0, d0 % vecSize), ValueRange{OC});
    Value maskVector = vector::CreateMaskOp::create(rewriter, 
        loc, VectorType::get({vecSize}, rewriter.getI1Type()),
        ValueRange{tailLength});

    Value FH = memref::DimOp::create(rewriter, loc, filter, 0); // FH
    Value FW = memref::DimOp::create(rewriter, loc, filter, 1); // FW

    // clang format off
    //  Step 1: Create outer most loops.
    // Create the scf::ForallOp operation For N,OH,OW
    scf::ForallOp::create(rewriter, 
        loc, SmallVector<OpFoldResult, 3>({N, OH, OW}), ValueRange{},
        std::nullopt, // No mapping specified in this example
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange loopIndices) {
          Value ivN = loopIndices[0];  // Index for the first dimension N
          Value ivOH = loopIndices[1]; // Index for the second dimension OH
          Value ivOW = loopIndices[2]; // Index for the third dimension OW
          // OC
          scf::ForOp::create(nestedBuilder, 
              nestedLoc, c0, applyOC, vecSizeValue, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value ivOC,
                  ValueRange iargs) {
                Value tVec = vector::LoadOp::create(builder, 
                    loc, vecTy, output, ValueRange{ivN, ivOH, ivOW, ivOC});

                // FH
                auto forOp = scf::ForOp::create(builder, 
                    loc, c0, FH, c1, ValueRange{tVec},
                    [&](OpBuilder &builder, Location loc, Value ivFH,
                        ValueRange iargs) {
                      Value rowInput = affine::AffineApplyOp::create(builder, 
                          loc,
                          AffineMap::get(2, 0, d0 * strHeight + d1 * dilHeight),
                          ValueRange{ivOH, ivFH});
                      Value rowFilter = ivFH;
                      // FW
                      auto forOp = scf::ForOp::create(builder, 
                          loc, c0, FW, c1, ValueRange{iargs[0]},
                          [&](OpBuilder &builder, Location loc, Value ivFW,
                              ValueRange iargs) {
                            Value columnInput =
                                affine::AffineApplyOp::create(builder, 
                                    loc,
                                    AffineMap::get(
                                        2, 0, d0 * strWidth + d1 * dilWidth),
                                    ValueRange{ivOW, ivFW});
                            Value columnFilter =
                                affine::AffineApplyOp::create(builder, 
                                    loc, AffineMap::get(1, 0, d0), ivFW);
                            Value iVec = vector::LoadOp::create(builder, 
                                loc, vecTy, input,
                                ValueRange{ivN, rowInput, columnInput, ivOC});
                            Value fVec = vector::LoadOp::create(builder, 
                                loc, vecTy, filter,
                                ValueRange{rowFilter, columnFilter, ivOC});
                            Value tVecNext;
                            if (isa<IntegerType>(elemTy)) {
                              Value mulVec = arith::MulIOp::create(builder, 
                                  loc, iVec, fVec);
                              tVecNext = arith::AddIOp::create(builder, 
                                  loc, mulVec, iargs[0]);
                            } else {
                              tVecNext = vector::FMAOp::create(builder, 
                                  loc, vecTy, iVec, fVec, iargs[0]);
                            }

                            scf::YieldOp::create(builder, loc,
                                                         ValueRange{tVecNext});
                          });
                      scf::YieldOp::create(builder, 
                          loc, ValueRange{forOp.getResult(0)});
                    });
                vector::StoreOp::create(builder, 
                    loc, forOp.getResult(0), output,
                    ValueRange{ivN, ivOH, ivOW, ivOC});

                scf::YieldOp::create(builder, loc, ValueRange{});
              });

          // applyOC
          Value condition = arith::CmpIOp::create(nestedBuilder, 
              loc, arith::CmpIPredicate::sgt, tailLength, c0);
          scf::IfOp::create(nestedBuilder, 
              loc, condition, [&](OpBuilder &builder, Location loc) {
                Value tVec = vector::MaskedLoadOp::create(builder, 
                    loc, vecTy, output, ValueRange{ivN, ivOH, ivOW, applyOC},
                    maskVector, zeroElementTypeVec);
                // FH
                auto forOp = scf::ForOp::create(builder, 
                    loc, c0, FH, c1, ValueRange{tVec},
                    [&](OpBuilder &builder, Location loc, Value ivFH,
                        ValueRange iargs) {
                      Value rowInput = affine::AffineApplyOp::create(builder, 
                          loc,
                          AffineMap::get(2, 0, d0 * strHeight + d1 * dilHeight),
                          ValueRange{ivOH, ivFH});
                      Value rowFilter = ivFH;
                      // FW
                      auto forOp = scf::ForOp::create(builder, 
                          loc, c0, FW, c1, ValueRange{iargs[0]},
                          [&](OpBuilder &builder, Location loc, Value ivFW,
                              ValueRange iargs) {
                            Value columnInput =
                                affine::AffineApplyOp::create(builder, 
                                    loc,
                                    AffineMap::get(
                                        2, 0, d0 * strWidth + d1 * dilWidth),
                                    ValueRange{ivOW, ivFW});
                            Value columnFilter =
                                affine::AffineApplyOp::create(builder, 
                                    loc, AffineMap::get(1, 0, d0), ivFW);
                            Value iVec = vector::MaskedLoadOp::create(builder, 
                                loc, vecTy, input,
                                ValueRange{ivN, rowInput, columnInput, applyOC},
                                maskVector, zeroElementTypeVec);
                            Value fVec = vector::MaskedLoadOp::create(builder, 
                                loc, vecTy, filter,
                                ValueRange{rowFilter, columnFilter, applyOC},
                                maskVector, zeroElementTypeVec);
                            Value tVecNext;
                            if (isa<IntegerType>(elemTy)) {
                              Value mulVec = arith::MulIOp::create(builder, 
                                  loc, iVec, fVec);
                              tVecNext = arith::AddIOp::create(builder, 
                                  loc, mulVec, iargs[0]);
                            } else {
                              tVecNext = vector::FMAOp::create(builder, 
                                  loc, vecTy, iVec, fVec, iargs[0]);
                            }

                            scf::YieldOp::create(builder, loc,
                                                         ValueRange{tVecNext});
                          });
                      scf::YieldOp::create(builder, 
                          loc, ValueRange{forOp.getResult(0)});
                    });
                vector::MaskedStoreOp::create(builder, 
                    loc, output, ValueRange{ivN, ivOH, ivOW, applyOC},
                    maskVector, forOp.getResult(0));
                scf::YieldOp::create(builder, loc, ValueRange{});
              });

          scf::InParallelOp::create(nestedBuilder, nestedLoc);
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
// DepthwiseConv2DNhwcHwcOptimizePass
//===----------------------------------------------------------------------===//

namespace {
class DepthwiseConv2DNhwcHwcOptimizePass
    : public PassWrapper<DepthwiseConv2DNhwcHwcOptimizePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      DepthwiseConv2DNhwcHwcOptimizePass)
  StringRef getArgument() const final {
    return "depthwise-conv-nhwc-hwc-optimize";
  }
  StringRef getDescription() const final {
    return "Depthwise Conv2d NHWC HWC optimize.";
  }
  DepthwiseConv2DNhwcHwcOptimizePass() = default;
  DepthwiseConv2DNhwcHwcOptimizePass(
      const DepthwiseConv2DNhwcHwcOptimizePass &) {}
  explicit DepthwiseConv2DNhwcHwcOptimizePass(int64_t vecSizeParam) {
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

void DepthwiseConv2DNhwcHwcOptimizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<DepthwiseConv2DNhwcHwcOptimizePattern>(context, vecSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerDepthwiseConv2DNhwcHwcOptimizePass() {
  PassRegistration<DepthwiseConv2DNhwcHwcOptimizePass>();
}
} // namespace buddy
} // namespace mlir
