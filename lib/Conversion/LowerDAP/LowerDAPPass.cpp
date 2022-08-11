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

class DAPBiquadLowering : public OpRewritePattern<dap::BiquadOp> {
public:
  using OpRewritePattern<dap::BiquadOp>::OpRewritePattern;

  explicit DAPBiquadLowering(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(dap::BiquadOp op, PatternRewriter &rewriter) const override {
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

    Value b0 = rewriter.create<memref:: LoadOp>(loc, kernel, ValueRange{c0});
    Value b1 = rewriter.create<memref:: LoadOp>(loc, kernel, ValueRange{c1});
    Value b2 = rewriter.create<memref:: LoadOp>(loc, kernel, ValueRange{c2});
    Value a0 = rewriter.create<memref:: LoadOp>(loc, kernel, ValueRange{c3});
    Value a1 = rewriter.create<memref:: LoadOp>(loc, kernel, ValueRange{c4});
    Value a2 = rewriter.create<memref:: LoadOp>(loc, kernel, ValueRange{c5});

    Value N = rewriter.create<memref:: DimOp>(loc, input, c0);

    FloatType f32 = FloatType::getF32(ctx);
    
    Value z1 = rewriter.create<ConstantFloatOp>(loc, APFloat(float(0)), f32);
    Value z2 = rewriter.create<ConstantFloatOp>(loc, APFloat(float(0)), f32);

    SmallVector<Value, 8> lowerBound{c0};
    SmallVector<Value, 8> upperBound{N};
    SmallVector<int64_t, 8> steps{1};

    // mlir::buildAffineLoopNest(
    //   rewriter, loc, lowerBound, upperBound, steps,
    //   [&](OpBuilder &builder, Location loc, ValueRange ivs) {
    //     Value in = builder.create<memref::LoadOp>(loc, input, ValueRange{ivs[0]});
    //     Value t0 = builder.create<MulFOp>(loc, b0, in);
    //     Value opt = builder.create<AddFOp>(loc, t0, z1);

    //     Value t1 = builder.create<MulFOp>(loc, b1, in);
    //     Value t2 = builder.create<MulFOp>(loc, a1, opt);
    //     Value t3 = builder.create<SubFOp>(loc, t1, t2);
    //     z1 = builder.create<AddFOp>(loc, z2, t3);

    //     Value t4 = builder.create<MulFOp>(loc, b2, in);
    //     Value t5 = builder.create<MulFOp>(loc, a2, opt);
    //     z2 = builder.create<SubFOp>(loc, t4, t5);

    //     builder.create<memref::StoreOp>(loc, opt, output, ValueRange{ivs[0]});
        
    //   });
   
    mlir::scf::buildLoopNest(rewriter, loc, ValueRange(c0),ValueRange(N),ValueRange(c1), ValueRange{z1, z2},
      [&](OpBuilder &builder, Location loc, ValueRange ivs, ValueRange ivc) -> scf::ValueVector {
      Value in = builder.create<memref::LoadOp>(loc, input, ValueRange{ivs[0]});
      Value t0 = builder.create<MulFOp>(loc, b0, in);
      Value opt = builder.create<AddFOp>(loc, t0, ivc[0]);

      Value t1 = builder.create<MulFOp>(loc, b1, in);
      Value t2 = builder.create<MulFOp>(loc, a1, opt);
      Value t3 = builder.create<SubFOp>(loc, t1, t2);
      Value z1_next = builder.create<AddFOp>(loc, ivc[1], t3);

      Value t4 = builder.create<MulFOp>(loc, b2, in);
      Value t5 = builder.create<MulFOp>(loc, a2, opt);
      Value z2_next = builder.create<SubFOp>(loc, t4, t5);

      builder.create<memref::StoreOp>(loc, opt, output, ValueRange{ivs[0]});
      return std::vector<Value>{z1_next, z2_next};
    });
    

    rewriter.eraseOp(op);
    return success();
  } 
};

} // end anonymous namespace

void populateLowerDAPConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<DAPFirLowering>(patterns.getContext());
  patterns.add<DAPBiquadLowering>(patterns.getContext());
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

  StringRef getArgument() const final { return "lower-dap"; }
  StringRef getDescription() const final { return "Lower DAP Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<buddy::dap::DAPDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect, VectorDialect,
                    AffineDialect, arith::ArithmeticDialect>();
  }
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
  populateLowerDAPConversionPatterns(patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerDAPPass() { PassRegistration<LowerDAPPass>(); }
} // namespace buddy
} // namespace mlir
