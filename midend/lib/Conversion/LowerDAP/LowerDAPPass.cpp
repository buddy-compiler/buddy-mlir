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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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
    auto ctx = op->getContext();
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    FloatType f32 = Float32Type::get(ctx);
    Value f0 =
        rewriter.create<arith::ConstantFloatOp>(loc, APFloat(float(0.0)), f32);

    Value kernelSize = rewriter.create<memref::DimOp>(loc, kernel, c0);
    Value dataLen = rewriter.create<memref::DimOp>(loc, output, c0);

    // Populate the FIR pipeline by padding the `input` with [`kernelSize`-1]
    // zeros at the beginning. Compute only the padding section of the input
    // data.
    Value fillInLen = rewriter.create<arith::SubIOp>(loc, kernelSize, c1);
    rewriter.create<scf::ForOp>(
        loc, c0, fillInLen, c1, ValueRange{std::nullopt},
        [&](OpBuilder &b, Location loc, Value iv_n, ValueRange iargs) {
          Value upperBound = b.create<arith::AddIOp>(loc, iv_n, c1);
          Value outFinal =
              b.create<scf::ForOp>(
                   loc, c0, upperBound, c1, ValueRange{f0},
                   [&](OpBuilder &b, Location loc, Value iv_k,
                       ValueRange iargs) {
                     Value i = b.create<arith::SubIOp>(loc, iv_n, iv_k);
                     Value in = b.create<memref::LoadOp>(loc, input, i);
                     Value k = b.create<memref::LoadOp>(loc, kernel, iv_k);
                     Value mul = b.create<arith::MulFOp>(loc, in, k);
                     Value outNext =
                         b.create<arith::AddFOp>(loc, iargs[0], mul);
                     b.create<scf::YieldOp>(loc, outNext);
                   })
                  .getResult(0);
          b.create<memref::StoreOp>(loc, outFinal, output, ValueRange{iv_n});
          b.create<scf::YieldOp>(loc, std::nullopt);
        });

    // Compute the input data following the padding section.
    rewriter.create<scf::ForOp>(
        loc, fillInLen, dataLen, c1, ValueRange{std::nullopt},
        [&](OpBuilder &b, Location loc, Value iv_n, ValueRange iargs) {
          Value outFinal =
              b.create<scf::ForOp>(
                   loc, c0, kernelSize, c1, ValueRange{f0},
                   [&](OpBuilder &b, Location loc, Value iv_k,
                       ValueRange iargs) {
                     Value i = b.create<arith::SubIOp>(loc, iv_n, iv_k);
                     Value in = b.create<memref::LoadOp>(loc, input, i);
                     Value k = b.create<memref::LoadOp>(loc, kernel, iv_k);
                     Value mul = b.create<arith::MulFOp>(loc, in, k);
                     Value outNext =
                         b.create<arith::AddFOp>(loc, iargs[0], mul);
                     b.create<scf::YieldOp>(loc, outNext);
                   })
                  .getResult(0);
          b.create<memref::StoreOp>(loc, outFinal, output, ValueRange{iv_n});
          b.create<scf::YieldOp>(loc, std::nullopt);
        });

    rewriter.eraseOp(op);
    return success();
  }
};

class DAPBiquadLowering : public OpRewritePattern<dap::BiquadOp> {
public:
  using OpRewritePattern<dap::BiquadOp>::OpRewritePattern;

  explicit DAPBiquadLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dap::BiquadOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);

    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<ConstantIndexOp>(loc, 2);
    Value c4 = rewriter.create<ConstantIndexOp>(loc, 4);
    Value c5 = rewriter.create<ConstantIndexOp>(loc, 5);

    Value b0 = rewriter.create<memref::LoadOp>(loc, kernel, ValueRange{c0});
    Value b1 = rewriter.create<memref::LoadOp>(loc, kernel, ValueRange{c1});
    Value b2 = rewriter.create<memref::LoadOp>(loc, kernel, ValueRange{c2});
    // Value a0 of kernel is not used
    Value a1 = rewriter.create<memref::LoadOp>(loc, kernel, ValueRange{c4});
    Value a2 = rewriter.create<memref::LoadOp>(loc, kernel, ValueRange{c5});

    Value N = rewriter.create<memref::DimOp>(loc, input, c0);

    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);

    FloatType f32 = Float32Type::get(ctx);

    Value z1 = rewriter.create<ConstantFloatOp>(loc, APFloat(float(0)), f32);
    Value z2 = rewriter.create<ConstantFloatOp>(loc, APFloat(float(0)), f32);

    VectorType vectorTy32 = VectorType::get({stride}, f32);

    Value x0 = rewriter.create<memref::LoadOp>(loc, input, ValueRange{c0});
    Value x = rewriter.create<MulFOp>(loc, b0, x0);
    rewriter.create<memref::StoreOp>(loc, x, output, ValueRange{c0});

    Value x1 = rewriter.create<memref::LoadOp>(loc, input, ValueRange{c1});
    Value x2 = rewriter.create<MulFOp>(loc, b0, x1);
    Value x3 = rewriter.create<MulFOp>(loc, b1, x0);
    Value x4 = rewriter.create<AddFOp>(loc, x2, x3);
    rewriter.create<memref::StoreOp>(loc, x4, output, ValueRange{c1});

    Value Vecb0 = rewriter.create<vector::BroadcastOp>(loc, vectorTy32, b0);
    Value Vecb1 = rewriter.create<vector::BroadcastOp>(loc, vectorTy32, b1);
    Value Vecb2 = rewriter.create<vector::BroadcastOp>(loc, vectorTy32, b2);

    // A biquad filter expression:
    // y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] + a1*y[n-1] + a2*y[n-2];
    // FIR part
    rewriter.create<scf::ForOp>(
        loc, c2, N, strideVal, ValueRange{std::nullopt},
        [&](OpBuilder &builder, Location loc, Value ivs, ValueRange iargs) {
          Value idx0 = ivs;
          Value idx1 = builder.create<SubIOp>(loc, idx0, c1);
          Value idx2 = builder.create<SubIOp>(loc, idx0, c2);

          Value inputVec0 =
              builder.create<LoadOp>(loc, vectorTy32, input, ValueRange{idx0});
          Value inputVec1 =
              builder.create<LoadOp>(loc, vectorTy32, input, ValueRange{idx1});
          Value inputVec2 =
              builder.create<LoadOp>(loc, vectorTy32, input, ValueRange{idx2});

          Value outputVec =
              builder.create<LoadOp>(loc, vectorTy32, output, ValueRange{idx0});
          Value resVec0 =
              builder.create<FMAOp>(loc, inputVec0, Vecb0, outputVec);
          Value resVec1 = builder.create<FMAOp>(loc, inputVec1, Vecb1, resVec0);
          Value resVec2 = builder.create<FMAOp>(loc, inputVec2, Vecb2, resVec1);
          builder.create<StoreOp>(loc, resVec2, output, ValueRange{idx0});
          builder.create<scf::YieldOp>(loc, std::nullopt);
        });

    // IIR part
    rewriter.create<scf::ForOp>(
        loc, c0, N, c1, ValueRange{z1, z2},
        [&](OpBuilder &builder, Location loc, Value ivs, ValueRange iargs) {
          Value x =
              builder.create<memref::LoadOp>(loc, output, ValueRange(ivs));
          Value t1 = builder.create<MulFOp>(loc, a1, iargs[1]);
          Value t2 = builder.create<MulFOp>(loc, a2, iargs[0]);
          Value y = builder.create<AddFOp>(loc, t1, t2);
          Value opt = builder.create<SubFOp>(loc, x, y);

          builder.create<memref::StoreOp>(loc, opt, output, ValueRange{ivs});

          builder.create<scf::YieldOp>(loc, std::vector<Value>{iargs[1], opt});
        });

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
};

class DAPIirLowering : public OpRewritePattern<dap::IirOp> {
public:
  using OpRewritePattern<dap::IirOp>::OpRewritePattern;

  explicit DAPIirLowering(MLIRContext *context) : OpRewritePattern(context) {}

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
    Value c4 = rewriter.create<ConstantIndexOp>(loc, 4);
    Value c5 = rewriter.create<ConstantIndexOp>(loc, 5);

    Value N = rewriter.create<memref::DimOp>(loc, input, c0);
    Value filterSize = rewriter.create<memref::DimOp>(loc, kernel, c0);

    FloatType f32 = Float32Type::get(ctx);

    // loop over every row in SOS matrix
    rewriter.create<scf::ForOp>(
        loc, c0, filterSize, c1, ValueRange{input},
        [&](OpBuilder &builder, Location loc, Value iv, ValueRange iarg) {
          Value b0 =
              builder.create<memref::LoadOp>(loc, kernel, ValueRange{iv, c0});
          Value b1 =
              builder.create<memref::LoadOp>(loc, kernel, ValueRange{iv, c1});
          Value b2 =
              builder.create<memref::LoadOp>(loc, kernel, ValueRange{iv, c2});
          Value a1 =
              builder.create<memref::LoadOp>(loc, kernel, ValueRange{iv, c4});
          Value a2 =
              builder.create<memref::LoadOp>(loc, kernel, ValueRange{iv, c5});

          Value z1 =
              builder.create<ConstantFloatOp>(loc, APFloat(float(0)), f32);
          Value z2 =
              builder.create<ConstantFloatOp>(loc, APFloat(float(0)), f32);

          // Loop reordering, compute z1 for next iteration, z2 for the second
          // following iteration.
          builder.create<scf::ForOp>(
              loc, c0, N, c1, ValueRange{z1, z2},
              [&](OpBuilder &builder, Location loc, Value iv,
                  ValueRange iargs) {
                Value inElem = builder.create<memref::LoadOp>(loc, iarg[0], iv);
                Value t0 = builder.create<arith::MulFOp>(loc, b0, inElem);
                Value outElem =
                    builder.create<arith::AddFOp>(loc, t0, iargs[0]);

                Value t1 = builder.create<arith::MulFOp>(loc, b1, inElem);
                Value t2 = builder.create<arith::MulFOp>(loc, a1, outElem);
                Value t3 = builder.create<arith::SubFOp>(loc, t1, t2);
                Value z1Next = builder.create<arith::AddFOp>(loc, t3, iargs[1]);

                Value t4 = builder.create<arith::MulFOp>(loc, b2, inElem);
                Value t5 = builder.create<arith::MulFOp>(loc, a2, outElem);
                Value z2Next = builder.create<arith::SubFOp>(loc, t4, t5);

                builder.create<memref::StoreOp>(loc, outElem, output, iv);
                builder.create<scf::YieldOp>(
                    loc, std::vector<Value>{z1Next, z2Next});
              });

          builder.create<scf::YieldOp>(loc, output);
        });

    rewriter.eraseOp(op);
    return success();
  }
};

} // end anonymous namespace

void populateLowerDAPConversionPatterns(RewritePatternSet &patterns,
                                        int64_t stride) {
  patterns.add<DAPFirLowering>(patterns.getContext());
  patterns.add<DAPBiquadLowering>(patterns.getContext(), stride);
  patterns.add<DAPIirLowering>(patterns.getContext());
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
                    affine::AffineDialect, arith::ArithDialect,
                    linalg::LinalgDialect>();
  }
  Option<int64_t> stride{*this, "DAP-vector-splitting",
                         llvm::cl::desc("Vector splitting size."),
                         llvm::cl::init(32)};
};
} // end anonymous namespace.

void LowerDAPPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<affine::AffineDialect, scf::SCFDialect,
                       func::FuncDialect, memref::MemRefDialect, VectorDialect,
                       arith::ArithDialect, linalg::LinalgDialect>();
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
