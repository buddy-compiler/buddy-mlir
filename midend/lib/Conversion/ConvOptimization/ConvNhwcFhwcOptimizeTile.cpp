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

class ConvNhwcFhwcTileOptimizePattern : public ConversionPattern {
public:
  explicit ConvNhwcFhwcTileOptimizePattern(MLIRContext *context,
                                           int64_t vecSizeParam,
                                           int64_t tilingOHParam,
                                           int64_t tilingOWParam)
      : ConversionPattern(linalg::Conv2DNhwcFhwcOp::getOperationName(), 1,
                          context) {
    vecSize = vecSizeParam;
    tilingOH = tilingOHParam;
    tilingOW = tilingOWParam;
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

    int strHeight, strWidth, dilHeight, dilWidth;

    // Strides.
    if (!convOp.getStrides()) {
      strHeight = 1;
      strWidth = 1;
    } else {
      strHeight = convOp.getStrides().getValues<int64_t>()[0];
      strWidth = convOp.getStrides().getValues<int64_t>()[1];
    }

    // Dilations.
    if (!convOp.getDilations()) {
      dilHeight = 1;
      dilWidth = 1;
    } else {
      dilHeight = convOp.getDilations().getValues<int64_t>()[0];
      dilWidth = convOp.getDilations().getValues<int64_t>()[1];
    }

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

    Value tilingOHLen = rewriter.create<affine::AffineApplyOp>(
        loc, AffineMap::get(1, 0, d0.ceilDiv(tilingOH) * tilingOH),
        ValueRange{OH});
    Value tilingOWLen = rewriter.create<affine::AffineApplyOp>(
        loc, AffineMap::get(1, 0, d0.ceilDiv(tilingOW) * tilingOW),
        ValueRange{OW});

    auto tilingUpperBounder =
        AffineMap::get(2, 1, {d0 + d1, s0}, rewriter.getContext());
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
    rewriter.create<scf::ForOp>(
        loc, c0, N, c1, ValueRange{std::nullopt},
        [&](OpBuilder &builder, Location loc, Value ivN, ValueRange iargs) {
          // OHT
          builder.create<scf::ForOp>(
              loc, c0, OH, tilingOHLen, ValueRange{std::nullopt},
              [&](OpBuilder &builder, Location loc, Value ivOHT,
                  ValueRange iargs) {
                Value ivOHTUpper = builder.create<affine::AffineMinOp>(
                    loc, tilingUpperBounder,
                    ValueRange{ivOHT, tilingOHLen, OH});
                // OH
                builder.create<scf::ForOp>(
                    loc, ivOHT, ivOHTUpper, c1, ValueRange{std::nullopt},
                    [&](OpBuilder &builder, Location loc, Value ivOH,
                        ValueRange iargs) {
                      // OWT
                      builder.create<scf::ForOp>(
                          loc, c0, OW, tilingOWLen, ValueRange{std::nullopt},
                          [&](OpBuilder &builder, Location loc, Value ivOWT,
                              ValueRange iargs) {
                            Value ivOWTUpper =
                                builder.create<affine::AffineMinOp>(
                                    loc, tilingUpperBounder,
                                    ValueRange{ivOWT, tilingOWLen, OW});
                            // OW
                            builder.create<scf::ForOp>(
                                loc, ivOWT, ivOWTUpper, c1,
                                ValueRange{std::nullopt},
                                [&](OpBuilder &builder, Location loc,
                                    Value ivOW, ValueRange iargs) {
                                  // OC
                                  builder.create<scf::ForOp>(
                                      loc, c0, OC, c1, ValueRange{std::nullopt},
                                      [&](OpBuilder &builder, Location loc,
                                          Value ivOC, ValueRange iargs) {
                                        Value addRes =
                                            builder.create<memref::LoadOp>(
                                                loc, output,
                                                ValueRange{ivN, ivOH, ivOW,
                                                           ivOC});
                                        // IC
                                        auto forOp = builder.create<scf::ForOp>(
                                            loc, c0, fixedIC, vecSizeValue,
                                            ValueRange{addRes},
                                            [&](OpBuilder &builder,
                                                Location loc, Value ivIC,
                                                ValueRange iargs) {
                                              Value tVec =
                                                  builder.create<SplatOp>(
                                                      loc, vecTy, cf0);
                                              Value remainLen = builder.create<
                                                  affine::AffineMinOp>(
                                                  loc,
                                                  AffineMap::get(
                                                      2, 1, {-d0 + s0, d1},
                                                      builder.getContext()),
                                                  ValueRange{ivIC, vecSizeValue,
                                                             IC});
                                              Value remainMask = builder.create<
                                                  vector::CreateMaskOp>(
                                                  loc,
                                                  VectorType::get(
                                                      {vecSize},
                                                      rewriter.getI1Type()),
                                                  ValueRange{remainLen});

                                              // FH
                                              auto forOp = builder.create<
                                                  scf::ForOp>(
                                                  loc, c0, FH, c1,
                                                  ValueRange{tVec},
                                                  [&](OpBuilder &builder,
                                                      Location loc, Value ivFH,
                                                      ValueRange iargs) {
                                                    Value rowInput =
                                                        builder.create<
                                                            affine::
                                                                AffineApplyOp>(
                                                            loc,
                                                            AffineMap::get(
                                                                2, 0,
                                                                d0 * strHeight +
                                                                    d1 *
                                                                        dilHeight),
                                                            ValueRange{ivOH,
                                                                       ivFH});
                                                    Value rowFilter = ivFH;
                                                    // FW
                                                    auto forOp = builder.create<
                                                        scf::ForOp>(
                                                        loc, c0, FW, c1,
                                                        ValueRange{iargs[0]},
                                                        [&](OpBuilder &builder,
                                                            Location loc,
                                                            Value ivFW,
                                                            ValueRange iargs) {
                                                          Value columnInput =
                                                              builder.create<
                                                                  affine::
                                                                      AffineApplyOp>(
                                                                  loc,
                                                                  AffineMap::get(
                                                                      2, 0,
                                                                      d0 * strWidth +
                                                                          d1 *
                                                                              dilWidth),
                                                                  ValueRange{
                                                                      ivOW,
                                                                      ivFW});
                                                          Value columnFilter =
                                                              builder.create<
                                                                  affine::
                                                                      AffineApplyOp>(
                                                                  loc,
                                                                  AffineMap::
                                                                      get(1, 0,
                                                                          d0),
                                                                  ivFW);
                                                          Value iVec =
                                                              builder.create<
                                                                  vector::
                                                                      LoadOp>(
                                                                  loc, vecTy,
                                                                  input,
                                                                  ValueRange{
                                                                      ivN,
                                                                      rowInput,
                                                                      columnInput,
                                                                      ivIC});
                                                          Value fVec =
                                                              builder.create<
                                                                  vector::
                                                                      LoadOp>(
                                                                  loc, vecTy,
                                                                  filter,
                                                                  ValueRange{
                                                                      ivOC,
                                                                      rowFilter,
                                                                      columnFilter,
                                                                      ivIC});
                                                          Value tVecNext =
                                                              builder.create<
                                                                  vector::
                                                                      FMAOp>(
                                                                  loc, vecTy,
                                                                  iVec, fVec,
                                                                  iargs[0]);
                                                          builder.create<
                                                              scf::YieldOp>(
                                                              loc,
                                                              ValueRange{
                                                                  tVecNext});
                                                        });
                                                    builder
                                                        .create<scf::YieldOp>(
                                                            loc,
                                                            ValueRange{
                                                                forOp.getResult(
                                                                    0)});
                                                  });
                                              auto reduceVecOp = builder.create<
                                                  vector::ReductionOp>(
                                                  loc,
                                                  vector::CombiningKind::ADD,
                                                  forOp.getResult(0));
                                              auto maskedOp = cast<
                                                  vector::MaskOp>(
                                                  mlir::vector::maskOperation(
                                                      builder, reduceVecOp,
                                                      remainMask));
                                              Value reduceVec =
                                                  maskedOp->getResult(0);
                                              Value addNext =
                                                  builder.create<arith::AddFOp>(
                                                      loc, iargs[0], reduceVec);
                                              builder.create<scf::YieldOp>(
                                                  loc, ValueRange{addNext});
                                              //   builder.create<vector::StoreOp>(
                                              //   loc, tVec, vecBuffer, ivIC);
                                            });

                                        builder.create<memref::StoreOp>(
                                            loc, forOp.getResult(0), output,
                                            ValueRange{ivN, ivOH, ivOW, ivOC});
                                        builder.create<scf::YieldOp>(
                                            loc, std::nullopt);
                                      });
                                  builder.create<scf::YieldOp>(loc,
                                                               std::nullopt);
                                });
                            builder.create<scf::YieldOp>(loc, std::nullopt);
                          });
                      builder.create<scf::YieldOp>(loc, std::nullopt);
                    });
                builder.create<scf::YieldOp>(loc, std::nullopt);
              });
          builder.create<scf::YieldOp>(loc, std::nullopt);
        });
    // clang format on

    rewriter.create<memref::DeallocOp>(loc, vecBuffer);

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t vecSize;
  int64_t tilingOH;
  int64_t tilingOW;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ConvNhwcFhwcTileOptimizePass
//===----------------------------------------------------------------------===//

namespace {
class ConvNhwcFhwcTileOptimizePass
    : public PassWrapper<ConvNhwcFhwcTileOptimizePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvNhwcFhwcTileOptimizePass)
  StringRef getArgument() const final { return "conv-nhwc-fhwc-tile-optimize"; }
  StringRef getDescription() const final {
    return "Conv2d NHWC FHWC optimize with Tile.";
  }
  ConvNhwcFhwcTileOptimizePass() = default;
  ConvNhwcFhwcTileOptimizePass(const ConvNhwcFhwcTileOptimizePass &) {}
  explicit ConvNhwcFhwcTileOptimizePass(int64_t vecSizeParam) {
    vecSize = vecSizeParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect>();
  }

  Option<int64_t> vecSize{*this, "vec-size", llvm::cl::desc("Vector size."),
                          llvm::cl::init(16)};
  Option<int64_t> tilingOH{*this, "tiling-height",
                           llvm::cl::desc("tiling the output height."),
                           llvm::cl::init(0)};
  Option<int64_t> tilingOW{*this, "tiling-width",
                           llvm::cl::desc("tiling the output width."),
                           llvm::cl::init(0)};
};
} // end anonymous namespace.

void ConvNhwcFhwcTileOptimizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<ConvNhwcFhwcTileOptimizePattern>(context, vecSize, tilingOH,
                                                tilingOW);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerConvNhwcFhwcTileOptimizePass() {
  PassRegistration<ConvNhwcFhwcTileOptimizePass>();
}
} // namespace buddy
} // namespace mlir
