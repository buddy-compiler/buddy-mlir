//====- LowerDAPPass.cpp - DAP Dialect Lowering Pass  ---------------------===//
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
// This file defines DAP dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "DAP/DAPDialect.h"
#include "DAP/DAPOps.h"

using namespace mlir;
using namespace buddy;
using namespace vector;
using namespace mlir::arith;
using namespace mlir::linalg;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class DAPFirLowering : public OpRewritePattern<dap::FirOp> {
public:
  using OpRewritePattern<dap::FirOp>::OpRewritePattern;

  explicit DAPFirLowering(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(dap::FirOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    // auto ctx = op->getContext();
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);

    rewriter.create<linalg::Conv1DOp>(loc, ValueRange{input, kernel},
                                      ValueRange{output});

    rewriter.eraseOp(op);
    return success();
  }
};

class DAPIirLowering : public OpRewritePattern<dap::IirOp> {
public:
  using OpRewritePattern<dap::IirOp>::OpRewritePattern;

  explicit DAPIirLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dap::IirOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);

    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<ConstantIndexOp>(loc, 2);
    Value c3 = rewriter.create<ConstantIndexOp>(loc, 3);
    Value c4 = rewriter.create<ConstantIndexOp>(loc, 4);
    Value c5 = rewriter.create<ConstantIndexOp>(loc, 5);

    Value N = rewriter.create<memref::DimOp>(loc, input, c0);
    Value filterSize = rewriter.create<memref::DimOp>(loc, kernel, c0);
    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);

    FloatType f32 = FloatType::getF32(ctx);

    VectorType vectorTy32 = VectorType::get({stride}, f32);

    Value zr = rewriter.create<ConstantFloatOp>(loc, APFloat(float(0)), f32);

    rewriter.create<scf::ForOp>(
        loc, c0, filterSize, c1, ValueRange{llvm::None},
        [&](OpBuilder &builder, Location loc, ValueRange ivs,
            ValueRange iargs) {
          Value b0 = builder.create<memref::LoadOp>(loc, kernel,
                                                    ValueRange{ivs[0], c0});
          Value b1 = builder.create<memref::LoadOp>(loc, kernel,
                                                    ValueRange{ivs[0], c1});
          Value b2 = builder.create<memref::LoadOp>(loc, kernel,
                                                    ValueRange{ivs[0], c2});
          Value a0 = builder.create<memref::LoadOp>(loc, kernel,
                                                    ValueRange{ivs[0], c3});
          Value a1 = builder.create<memref::LoadOp>(loc, kernel,
                                                    ValueRange{ivs[0], c4});
          Value a2 = builder.create<memref::LoadOp>(loc, kernel,
                                                    ValueRange{ivs[0], c5});

          Value z1 =
              builder.create<ConstantFloatOp>(loc, APFloat(float(0)), f32);
          Value z2 =
              builder.create<ConstantFloatOp>(loc, APFloat(float(0)), f32);

          Value x0 = builder.create<memref::LoadOp>(loc, input, ValueRange{c0});
          Value temp = builder.create<MulFOp>(loc, b0, x0);
          builder.create<memref::StoreOp>(loc, temp, output, ValueRange{c0});

          Value x1 = builder.create<memref::LoadOp>(loc, input, ValueRange{c1});
          Value temp0 = builder.create<MulFOp>(loc, b0, x1);
          Value temp1 = builder.create<MulFOp>(loc, b1, x0);
          Value temp2 = builder.create<AddFOp>(loc, temp0, temp1);
          builder.create<memref::StoreOp>(loc, temp2, output, ValueRange{c1});

          Value Vecb0 = builder.create<BroadcastOp>(loc, vectorTy32, b0);
          Value Vecb1 = builder.create<BroadcastOp>(loc, vectorTy32, b1);
          Value Vecb2 = builder.create<BroadcastOp>(loc, vectorTy32, b2);

          builder.create<scf::ForOp>(
              loc, c2, N, strideVal, ValueRange{llvm::None},
              [&](OpBuilder &builder, Location loc, Value iv,
                  ValueRange itrargs) {
                Value idx0 = iv;
                Value idx1 = builder.create<SubIOp>(loc, idx0, c1);
                Value idx2 = builder.create<SubIOp>(loc, idx0, c2);

                Value inputVec0 = builder.create<LoadOp>(loc, vectorTy32, input,
                                                         ValueRange{idx0});
                Value inputVec1 = builder.create<LoadOp>(loc, vectorTy32, input,
                                                         ValueRange{idx1});
                Value inputVec2 = builder.create<LoadOp>(loc, vectorTy32, input,
                                                         ValueRange{idx2});

                Value outputVec =
                    rewriter.create<BroadcastOp>(loc, vectorTy32, zr);
                Value resVec0 =
                    builder.create<FMAOp>(loc, inputVec0, Vecb0, outputVec);
                Value resVec1 =
                    builder.create<FMAOp>(loc, inputVec1, Vecb1, resVec0);
                Value resVec2 =
                    builder.create<FMAOp>(loc, inputVec2, Vecb2, resVec1);
                builder.create<StoreOp>(loc, resVec2, output, ValueRange{idx0});

                builder.create<scf::YieldOp>(loc, llvm::None);
              });

          builder.create<scf::ForOp>(
              loc, c0, N, c1, ValueRange{z1, z2},
              [&](OpBuilder &builder, Location loc, Value iv,
                  ValueRange itrargs) {
                Value x =
                    builder.create<memref::LoadOp>(loc, output, ValueRange{iv});
                Value t1 = builder.create<MulFOp>(loc, a1, itrargs[1]);
                Value t2 = builder.create<MulFOp>(loc, a2, itrargs[0]);
                Value y = builder.create<AddFOp>(loc, t1, t2);
                Value opt = builder.create<SubFOp>(loc, x, y);

                builder.create<memref::StoreOp>(loc, opt, output,
                                                ValueRange{iv});
                builder.create<scf::YieldOp>(
                    loc, std::vector<Value>{itrargs[1], opt});
              });
          builder.create<memref::CopyOp>(loc, output, input);
          builder.create<scf::YieldOp>(loc, llvm::None);
        });

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
};
} // end anonymous namespace

void populateLowerDAPConversionPatterns(RewritePatternSet &patterns,
                                        int64_t stride) {
  patterns.add<DAPFirLowering>(patterns.getContext());
  patterns.add<DAPIirLowering>(patterns.getContext(), stride);
}

//===----------------------------------------------------------------------===//
// LowerDAPPass
//===----------------------------------------------------------------------===//

namespace {
class LowerDAPPass : public PassWrapper<LowerDAPPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerDAPPass)
  LowerDAPPass() = default;
  LowerDAPPass(const LowerDAPPass &) {}
  explicit LowerDAPPass(int64_t strideParam) { stride = strideParam; }

  StringRef getArgument() const final { return "lower-dap"; }
  StringRef getDescription() const final { return "Lower DAP Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<buddy::dap::DAPDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect, VectorDialect,
                    AffineDialect, arith::ArithmeticDialect>();
  }
  Option<int64_t> stride{*this, "DAP-strip-mining",
                         llvm::cl::desc("Strip mining size."),
                         llvm::cl::init(32)};
};
} // end anonymous namespace.

void LowerDAPPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<AffineDialect, scf::SCFDialect, func::FuncDialect,
                         memref::MemRefDialect, VectorDialect,
                         arith::ArithmeticDialect, linalg::LinalgDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();

  RewritePatternSet patterns(context);
  populateLowerDAPConversionPatterns(patterns, stride);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerDAPPass() { PassRegistration<LowerDAPPass>(); }
} // namespace buddy
} // namespace mlir
