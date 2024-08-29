//====- ConvOptimize.cpp --------------------------------------------------===//
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
    auto loc = op->getLoc();

    // Some constant we need.
    const Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    const Value cf0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.));

    const Value vecSizeValue =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(vecSize));

    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);
    const AffineExpr s0 = rewriter.getAffineSymbolExpr(0);

    Value input = op->getOperand(0);
    Value filter = op->getOperand(1);
    Value output = op->getOperand(2);

    ShapedType inputTy = input.getType().cast<ShapedType>();

    Type elemTy = inputTy.getElementType();
    VectorType vecTy = VectorType::get(vecSize, elemTy);

    // Dims
    Value N = rewriter.create<memref::DimOp>(loc, output, 0);  // N
    Value OH = rewriter.create<memref::DimOp>(loc, output, 1); // OH
    Value OW = rewriter.create<memref::DimOp>(loc, output, 2); // OW
    Value OC = rewriter.create<memref::DimOp>(loc, output, 3); // OC
    Value IC = rewriter.create<memref::DimOp>(loc, input, 3);  // IC
    Value FH = rewriter.create<memref::DimOp>(loc, filter, 1); // FH
    Value FW = rewriter.create<memref::DimOp>(loc, filter, 2); // FW

    // memref<1xIC>
    Value fixedIC = rewriter.create<affine::AffineApplyOp>(
        loc, AffineMap::get(1, 0, d0.ceilDiv(vecSize) * vecSize),
        ValueRange{IC});
    MemRefType bufferTy = MemRefType::get(ShapedType::kDynamic, elemTy);
    Value vecBuffer =
        rewriter.create<memref::AllocOp>(loc, bufferTy, ValueRange{fixedIC});

    // clang format off
    //  Step 1: Create outer most loops.
    //  N
    affine::buildAffineLoopNest(
        rewriter, loc, c0, N, 1,
        [&](OpBuilder &, Location loc, ValueRange ivRange) {
          Value ivN = ivRange.front();
          // OH

          affine::buildAffineLoopNest(
              rewriter, loc, c0, OH, 1,
              [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                Value ivOH = ivRange.front();
                // OW
                affine::buildAffineLoopNest(
                    rewriter, loc, c0, OW, 1,
                    [&](OpBuilder &, Location loc, ValueRange ivRange) {
                      Value ivOW = ivRange.front();
                      // OC
                      affine::buildAffineLoopNest(
                          rewriter, loc, c0, OC, 1,
                          [&](OpBuilder &, Location loc, ValueRange ivRange) {
                            Value ivOC = ivRange.front();
                            Value addRes = builder.create<memref::LoadOp>(
                                loc, output, ValueRange{ivN, ivOH, ivOW, ivOC});
                            // IC
                            builder.create<scf::ForOp>(
                                loc, c0, fixedIC, vecSizeValue,
                                ValueRange{addRes},
                                [&](OpBuilder &builder, Location loc,
                                    Value ivIC, ValueRange iargs) {
                                  Value tVec =
                                      builder.create<SplatOp>(loc, vecTy, cf0);
                                  Value remainLen =
                                      builder.create<affine::AffineMinOp>(
                                          loc,
                                          AffineMap::get(2, 1, {-d0 + s0, d1},
                                                         builder.getContext()),
                                          ValueRange{ivIC, vecSizeValue, IC});
                                  Value remainMask =
                                      builder.create<vector::CreateMaskOp>(
                                          loc,
                                          VectorType::get({vecSize},
                                                          rewriter.getI1Type()),
                                          ValueRange{remainLen});

                                  // FH
                                  builder.create<scf::ForOp>(
                                      loc, c0, FH, c1, ValueRange{tVec},
                                      [&](OpBuilder &builder, Location loc,
                                          Value ivFH, ValueRange iargs) {
                                        Value rowInput =
                                            builder
                                                .create<affine::AffineApplyOp>(
                                                    loc,
                                                    AffineMap::get(2, 0,
                                                                   d0 + d1),
                                                    ValueRange{ivOH, ivFH});
                                        Value rowFilter = ivFH;
                                        // FW
                                        builder.create<scf::ForOp>(
                                            loc, c0, FW, c1,
                                            ValueRange{iargs[0]},
                                            [&](OpBuilder &builder,
                                                Location loc, Value ivFW,
                                                ValueRange iargs) {
                                              Value columnInput =
                                                  builder.create<
                                                      affine::AffineApplyOp>(
                                                      loc,
                                                      AffineMap::get(2, 0,
                                                                     d0 + d1),
                                                      ValueRange{ivOW, ivFW});
                                              Value columnFilter =
                                                  builder.create<
                                                      affine::AffineApplyOp>(
                                                      loc,
                                                      AffineMap::get(1, 0, d0),
                                                      ivFW);
                                              Value iVec =
                                                  builder
                                                      .create<vector::LoadOp>(
                                                          loc, vecTy, input,
                                                          ValueRange{
                                                              ivN, rowInput,
                                                              columnInput,
                                                              ivIC});
                                              Value fVec =
                                                  builder
                                                      .create<vector::LoadOp>(
                                                          loc, vecTy, filter,
                                                          ValueRange{
                                                              ivN, rowFilter,
                                                              columnFilter,
                                                              ivIC});
                                              Value tVec =
                                                  builder.create<vector::FMAOp>(
                                                      loc, vecTy, iVec, fVec,
                                                      iargs[0]);
                                              builder.create<scf::YieldOp>(
                                                  loc, ValueRange{tVec});
                                            });
                                        builder.create<scf::YieldOp>(
                                            loc, ValueRange{tVec});
                                      });
                                  auto reduceVecOp =
                                      builder.create<vector::ReductionOp>(
                                          loc, vector::CombiningKind::ADD,
                                          tVec);
                                  auto maskedOp = cast<vector::MaskOp>(
                                      mlir::vector::maskOperation(
                                          builder, reduceVecOp, remainMask));
                                  Value reduceVec = maskedOp->getResult(0);
                                  iargs[0] = builder.create<arith::AddFOp>(
                                      loc, iargs[0], reduceVec);
                                  builder.create<scf::YieldOp>(
                                      loc, ValueRange{iargs[0]});
                                  //   builder.create<vector::StoreOp>(
                                  //   loc, tVec, vecBuffer, ivIC);
                                });

                            builder.create<memref::StoreOp>(
                                loc, addRes, output,
                                ValueRange{ivN, ivOC, ivOH, ivOW});
                          });
                    });
              });
        });
    // clang format on

    rewriter.create<memref::DeallocOp>(loc, vecBuffer);

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

  Option<int64_t> vecSize{*this, "vec-size",
                          llvm::cl::desc("Vector size using in kernel."),
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
