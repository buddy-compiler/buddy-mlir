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

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    FloatType f32 = Float32Type::get(ctx);
    Value f0 =
        arith::ConstantFloatOp::create(rewriter, loc, f32, APFloat(float(0.0)));

    Value kernelSize = memref::DimOp::create(rewriter, loc, kernel, c0);
    Value dataLen = memref::DimOp::create(rewriter, loc, output, c0);

    // Populate the FIR pipeline by padding the `input` with [`kernelSize`-1]
    // zeros at the beginning. Compute only the padding section of the input
    // data.
    Value fillInLen = arith::SubIOp::create(rewriter, loc, kernelSize, c1);
    scf::ForOp::create(rewriter, 
        loc, c0, fillInLen, c1, ValueRange{},
        [&](OpBuilder &b, Location loc, Value iv_n, ValueRange iargs) {
          Value upperBound = arith::AddIOp::create(b, loc, iv_n, c1);
          Value outFinal =
              scf::ForOp::create(b, 
                   loc, c0, upperBound, c1, ValueRange{f0},
                   [&](OpBuilder &b, Location loc, Value iv_k,
                       ValueRange iargs) {
                     Value i = arith::SubIOp::create(b, loc, iv_n, iv_k);
                     Value in = memref::LoadOp::create(b, loc, input, i);
                     Value k = memref::LoadOp::create(b, loc, kernel, iv_k);
                     Value mul = arith::MulFOp::create(b, loc, in, k);
                     Value outNext =
                         arith::AddFOp::create(b, loc, iargs[0], mul);
                     scf::YieldOp::create(b, loc, outNext);
                   })
                  .getResult(0);
          memref::StoreOp::create(b, loc, outFinal, output, ValueRange{iv_n});
          scf::YieldOp::create(b, loc);
        });

    // Compute the input data following the padding section.
    scf::ForOp::create(rewriter, 
        loc, fillInLen, dataLen, c1, ValueRange{},
        [&](OpBuilder &b, Location loc, Value iv_n, ValueRange iargs) {
          Value outFinal =
              scf::ForOp::create(b, 
                   loc, c0, kernelSize, c1, ValueRange{f0},
                   [&](OpBuilder &b, Location loc, Value iv_k,
                       ValueRange iargs) {
                     Value i = arith::SubIOp::create(b, loc, iv_n, iv_k);
                     Value in = memref::LoadOp::create(b, loc, input, i);
                     Value k = memref::LoadOp::create(b, loc, kernel, iv_k);
                     Value mul = arith::MulFOp::create(b, loc, in, k);
                     Value outNext =
                         arith::AddFOp::create(b, loc, iargs[0], mul);
                     scf::YieldOp::create(b, loc, outNext);
                   })
                  .getResult(0);
          memref::StoreOp::create(b, loc, outFinal, output, ValueRange{iv_n});
          scf::YieldOp::create(b, loc);
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

    Value c0 = ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = ConstantIndexOp::create(rewriter, loc, 1);
    Value c2 = ConstantIndexOp::create(rewriter, loc, 2);
    Value c4 = ConstantIndexOp::create(rewriter, loc, 4);
    Value c5 = ConstantIndexOp::create(rewriter, loc, 5);

    Value b0 = memref::LoadOp::create(rewriter, loc, kernel, ValueRange{c0});
    Value b1 = memref::LoadOp::create(rewriter, loc, kernel, ValueRange{c1});
    Value b2 = memref::LoadOp::create(rewriter, loc, kernel, ValueRange{c2});
    // Value a0 of kernel is not used
    Value a1 = memref::LoadOp::create(rewriter, loc, kernel, ValueRange{c4});
    Value a2 = memref::LoadOp::create(rewriter, loc, kernel, ValueRange{c5});

    Value N = memref::DimOp::create(rewriter, loc, input, c0);

    Value strideVal = ConstantIndexOp::create(rewriter, loc, stride);

    FloatType f32 = Float32Type::get(ctx);

    Value z1 = ConstantFloatOp::create(rewriter, loc, f32, APFloat(float(0)));
    Value z2 = ConstantFloatOp::create(rewriter, loc, f32, APFloat(float(0)));

    VectorType vectorTy32 = VectorType::get({stride}, f32);

    Value x0 = memref::LoadOp::create(rewriter, loc, input, ValueRange{c0});
    Value x = MulFOp::create(rewriter, loc, b0, x0);
    memref::StoreOp::create(rewriter, loc, x, output, ValueRange{c0});

    Value x1 = memref::LoadOp::create(rewriter, loc, input, ValueRange{c1});
    Value x2 = MulFOp::create(rewriter, loc, b0, x1);
    Value x3 = MulFOp::create(rewriter, loc, b1, x0);
    Value x4 = AddFOp::create(rewriter, loc, x2, x3);
    memref::StoreOp::create(rewriter, loc, x4, output, ValueRange{c1});

    Value Vecb0 = vector::BroadcastOp::create(rewriter, loc, vectorTy32, b0);
    Value Vecb1 = vector::BroadcastOp::create(rewriter, loc, vectorTy32, b1);
    Value Vecb2 = vector::BroadcastOp::create(rewriter, loc, vectorTy32, b2);

    // A biquad filter expression:
    // y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] + a1*y[n-1] + a2*y[n-2];
    // FIR part
    scf::ForOp::create(rewriter, 
        loc, c2, N, strideVal, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value ivs, ValueRange iargs) {
          Value idx0 = ivs;
          Value idx1 = SubIOp::create(builder, loc, idx0, c1);
          Value idx2 = SubIOp::create(builder, loc, idx0, c2);

          Value inputVec0 =
              LoadOp::create(builder, loc, vectorTy32, input, ValueRange{idx0});
          Value inputVec1 =
              LoadOp::create(builder, loc, vectorTy32, input, ValueRange{idx1});
          Value inputVec2 =
              LoadOp::create(builder, loc, vectorTy32, input, ValueRange{idx2});

          Value outputVec =
              LoadOp::create(builder, loc, vectorTy32, output, ValueRange{idx0});
          Value resVec0 =
              FMAOp::create(builder, loc, inputVec0, Vecb0, outputVec);
          Value resVec1 = FMAOp::create(builder, loc, inputVec1, Vecb1, resVec0);
          Value resVec2 = FMAOp::create(builder, loc, inputVec2, Vecb2, resVec1);
          StoreOp::create(builder, loc, resVec2, output, ValueRange{idx0});
          scf::YieldOp::create(builder, loc);
        });

    // IIR part
    scf::ForOp::create(rewriter, 
        loc, c0, N, c1, ValueRange{z1, z2},
        [&](OpBuilder &builder, Location loc, Value ivs, ValueRange iargs) {
          Value x =
              memref::LoadOp::create(builder, loc, output, ValueRange(ivs));
          Value t1 = MulFOp::create(builder, loc, a1, iargs[1]);
          Value t2 = MulFOp::create(builder, loc, a2, iargs[0]);
          Value y = AddFOp::create(builder, loc, t1, t2);
          Value opt = SubFOp::create(builder, loc, x, y);

          memref::StoreOp::create(builder, loc, opt, output, ValueRange{ivs});

          scf::YieldOp::create(builder, loc, std::vector<Value>{iargs[1], opt});
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

    Value c0 = ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = ConstantIndexOp::create(rewriter, loc, 1);
    Value c2 = ConstantIndexOp::create(rewriter, loc, 2);
    Value c4 = ConstantIndexOp::create(rewriter, loc, 4);
    Value c5 = ConstantIndexOp::create(rewriter, loc, 5);

    Value N = memref::DimOp::create(rewriter, loc, input, c0);
    Value filterSize = memref::DimOp::create(rewriter, loc, kernel, c0);

    FloatType f32 = Float32Type::get(ctx);

    // loop over every row in SOS matrix
    scf::ForOp::create(rewriter, 
        loc, c0, filterSize, c1, ValueRange{input},
        [&](OpBuilder &builder, Location loc, Value iv, ValueRange iarg) {
          Value b0 =
              memref::LoadOp::create(builder, loc, kernel, ValueRange{iv, c0});
          Value b1 =
              memref::LoadOp::create(builder, loc, kernel, ValueRange{iv, c1});
          Value b2 =
              memref::LoadOp::create(builder, loc, kernel, ValueRange{iv, c2});
          Value a1 =
              memref::LoadOp::create(builder, loc, kernel, ValueRange{iv, c4});
          Value a2 =
              memref::LoadOp::create(builder, loc, kernel, ValueRange{iv, c5});

          Value z1 =
              ConstantFloatOp::create(builder, loc, f32, APFloat(float(0)));
          Value z2 =
              ConstantFloatOp::create(builder, loc, f32, APFloat(float(0)));

          // Loop reordering, compute z1 for next iteration, z2 for the second
          // following iteration.
          scf::ForOp::create(builder, 
              loc, c0, N, c1, ValueRange{z1, z2},
              [&](OpBuilder &builder, Location loc, Value iv,
                  ValueRange iargs) {
                Value inElem = memref::LoadOp::create(builder, loc, iarg[0], iv);
                Value t0 = arith::MulFOp::create(builder, loc, b0, inElem);
                Value outElem =
                    arith::AddFOp::create(builder, loc, t0, iargs[0]);

                Value t1 = arith::MulFOp::create(builder, loc, b1, inElem);
                Value t2 = arith::MulFOp::create(builder, loc, a1, outElem);
                Value t3 = arith::SubFOp::create(builder, loc, t1, t2);
                Value z1Next = arith::AddFOp::create(builder, loc, t3, iargs[1]);

                Value t4 = arith::MulFOp::create(builder, loc, b2, inElem);
                Value t5 = arith::MulFOp::create(builder, loc, a2, outElem);
                Value z2Next = arith::SubFOp::create(builder, loc, t4, t5);

                memref::StoreOp::create(builder, loc, outElem, output, iv);
                scf::YieldOp::create(builder, 
                    loc, std::vector<Value>{z1Next, z2Next});
              });

          scf::YieldOp::create(builder, loc, output);
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
