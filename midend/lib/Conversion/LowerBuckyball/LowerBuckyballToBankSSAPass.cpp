//====- LowerBuckyballToBankSSAPass.cpp - Expand to bank-SSA ops -----------===//
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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace buddy;

namespace {

static Value cstI64(OpBuilder &b, Location loc, uint64_t v) {
  return b.create<arith::ConstantOp>(loc, b.getI64Type(), b.getI64IntegerAttr(v));
}

static LogicalResult getStaticRowStrideDiv16(MemRefType ty, uint64_t &out) {
  SmallVector<int64_t, 4> strides;
  int64_t off = 0;
  if (failed(ty.getStridesAndOffset(strides, off)) || strides.size() < 2)
    return failure();
  if (ShapedType::isDynamic(strides[0]) || strides[0] <= 0 || strides[0] % 16 != 0)
    return failure();
  if (ShapedType::isDynamic(strides[1]) || strides[1] != 1)
    return failure();
  out = static_cast<uint64_t>(strides[0] / 16);
  return success();
}

class MatMulToBankSSAPattern : public OpRewritePattern<buckyball::MatMulOp> {
public:
  using OpRewritePattern<buckyball::MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(buckyball::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value aMem = op.getAMemArray();
    Value bMem = op.getBMemArray();
    Value cMem = op.getCMemArray();

    auto aTy = dyn_cast<MemRefType>(aMem.getType());
    auto bTy = dyn_cast<MemRefType>(bMem.getType());
    auto cTy = dyn_cast<MemRefType>(cMem.getType());
    if (!aTy || !bTy || !cTy || !aTy.hasStaticShape() || !bTy.hasStaticShape() ||
        !cTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "bb_matmul requires static rank-2 memrefs");

    uint64_t m = aTy.getShape()[0];
    uint64_t k = aTy.getShape()[1];
    uint64_t kb = bTy.getShape()[0];
    uint64_t n = bTy.getShape()[1];
    if (k != kb)
      return rewriter.notifyMatchFailure(op, "inner dimensions must match");
    if (k % 16 != 0 || n % 16 != 0)
      return rewriter.notifyMatchFailure(op, "K and N must be multiples of 16");

    uint64_t strideA = 1;
    uint64_t strideB = 0;
    uint64_t strideC = 0;
    if (failed(getStaticRowStrideDiv16(bTy, strideB)))
      return rewriter.notifyMatchFailure(op, "B requires static strided<[row,1]> and row % 16 == 0");
    if (failed(getStaticRowStrideDiv16(cTy, strideC)))
      return rewriter.notifyMatchFailure(op, "C requires static strided<[row,1]> and row % 16 == 0");

    uint64_t depthA = m * (k / 16);
    uint64_t depthB = k * (n / 16);
    uint64_t depthC = m * (n / 16);

    auto a0 = rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    auto b0 = rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    auto c0 = rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    c0->setAttr("col", rewriter.getI64IntegerAttr(4));

    auto a1 = rewriter.create<buckyball::BankMvinOp>(
        loc, rewriter.getI64Type(), aMem, a0.getBank(),
        cstI64(rewriter, loc, depthA), cstI64(rewriter, loc, strideA));
    auto b1 = rewriter.create<buckyball::BankMvinOp>(
        loc, rewriter.getI64Type(), bMem, b0.getBank(),
        cstI64(rewriter, loc, depthB), cstI64(rewriter, loc, strideB));
    auto c1 = rewriter.create<buckyball::BankMulWarp16Op>(
        loc, rewriter.getI64Type(), a1.getBankOut(), b1.getBankOut(), c0.getBank(),
        cstI64(rewriter, loc, k), cstI64(rewriter, loc, 0));
    auto c2 = rewriter.create<buckyball::BankMvoutOp>(
        loc, rewriter.getI64Type(), cMem, c1.getWrBankOut(),
        cstI64(rewriter, loc, depthC), cstI64(rewriter, loc, strideC));

    rewriter.create<buckyball::BankReleaseOp>(loc, a1.getBankOut());
    rewriter.create<buckyball::BankReleaseOp>(loc, b1.getBankOut());
    rewriter.create<buckyball::BankReleaseOp>(loc, c2.getBankOut());

    rewriter.eraseOp(op);
    return success();
  }
};

class TransposeToBankSSAPattern
    : public OpRewritePattern<buckyball::TransposeOp> {
public:
  using OpRewritePattern<buckyball::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(buckyball::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();
    auto inTy = dyn_cast<MemRefType>(input.getType());
    if (!inTy || !inTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "transpose needs static shapes");
    uint64_t rows = inTy.getShape()[0];
    uint64_t cols = inTy.getShape()[1];
    if (rows != 16 || cols % 16 != 0 || rows % 16 != 0)
      return rewriter.notifyMatchFailure(op, "transpose expects 16xK with K multiple of 16");

    uint64_t depthIn = rows * (cols / 16);
    uint64_t depthOut = cols * (rows / 16);
    Value stride1 = cstI64(rewriter, loc, 1);

    auto in0 = rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    auto out0 = rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    auto in1 = rewriter.create<buckyball::BankMvinOp>(
        loc, rewriter.getI64Type(), input, in0.getBank(),
        cstI64(rewriter, loc, depthIn), stride1);
    auto out1 = rewriter.create<buckyball::BankTransposeOp>(
        loc, rewriter.getI64Type(), in1.getBankOut(), out0.getBank(),
        cstI64(rewriter, loc, cols), cstI64(rewriter, loc, 0));
    auto out2 = rewriter.create<buckyball::BankMvoutOp>(
        loc, rewriter.getI64Type(), output, out1.getOutBankOut(),
        cstI64(rewriter, loc, depthOut), stride1);

    rewriter.create<buckyball::BankReleaseOp>(loc, in1.getBankOut());
    rewriter.create<buckyball::BankReleaseOp>(loc, out2.getBankOut());
    rewriter.eraseOp(op);
    return success();
  }
};

class Im2colToBankSSAPattern : public OpRewritePattern<buckyball::Im2colOp> {
public:
  using OpRewritePattern<buckyball::Im2colOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(buckyball::Im2colOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();
    auto inTy = dyn_cast<MemRefType>(input.getType());
    auto outTy = dyn_cast<MemRefType>(output.getType());
    if (!inTy || !outTy || !inTy.hasStaticShape() || !outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "im2col needs static shapes");

    uint64_t inRows = inTy.getShape()[0];
    uint64_t outRows = outTy.getShape()[0];
    Value stride1 = cstI64(rewriter, loc, 1);

    auto in0 = rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    auto out0 = rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    auto in1 = rewriter.create<buckyball::BankMvinOp>(
        loc, rewriter.getI64Type(), input, in0.getBank(),
        cstI64(rewriter, loc, inRows), stride1);
    auto out1 = rewriter.create<buckyball::BankIm2colOp>(
        loc, rewriter.getI64Type(), in1.getBankOut(), out0.getBank(), op.getKRow(),
        op.getKCol(), op.getInRow(), op.getInCol(), op.getStartRow(),
        op.getStartCol());
    auto out2 = rewriter.create<buckyball::BankMvoutOp>(
        loc, rewriter.getI64Type(), output, out1.getOutBankOut(),
        cstI64(rewriter, loc, outRows), stride1);

    rewriter.create<buckyball::BankReleaseOp>(loc, in1.getBankOut());
    rewriter.create<buckyball::BankReleaseOp>(loc, out2.getBankOut());
    rewriter.eraseOp(op);
    return success();
  }
};

class QuantToBankSSAPattern : public OpRewritePattern<buckyball::QuantOp> {
public:
  using OpRewritePattern<buckyball::QuantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(buckyball::QuantOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();
    auto inTy = dyn_cast<MemRefType>(input.getType());
    auto outTy = dyn_cast<MemRefType>(output.getType());
    if (!inTy || !outTy || !inTy.hasStaticShape() || !outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "quant needs static shapes");

    uint64_t rows = inTy.getShape()[0];
    uint64_t outRows = outTy.getShape()[0];
    Value stride1 = cstI64(rewriter, loc, 1);

    auto in0 = rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    in0->setAttr("row", rewriter.getI64IntegerAttr(4));
    auto out0 = rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    auto in1 = rewriter.create<buckyball::BankMvinOp>(
        loc, rewriter.getI64Type(), input, in0.getBank(),
        cstI64(rewriter, loc, rows), stride1);
    auto out1 = rewriter.create<buckyball::BankQuantOp>(
        loc, rewriter.getI64Type(), in1.getBankOut(), out0.getBank(),
        cstI64(rewriter, loc, rows), op.getScale());
    auto out2 = rewriter.create<buckyball::BankMvoutOp>(
        loc, rewriter.getI64Type(), output, out1.getOutBankOut(),
        cstI64(rewriter, loc, outRows), stride1);

    rewriter.create<buckyball::BankReleaseOp>(loc, in1.getBankOut());
    rewriter.create<buckyball::BankReleaseOp>(loc, out2.getBankOut());
    rewriter.eraseOp(op);
    return success();
  }
};

class DequantToBankSSAPattern : public OpRewritePattern<buckyball::DequantOp> {
public:
  using OpRewritePattern<buckyball::DequantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(buckyball::DequantOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();
    auto inTy = dyn_cast<MemRefType>(input.getType());
    auto outTy = dyn_cast<MemRefType>(output.getType());
    if (!inTy || !outTy || !inTy.hasStaticShape() || !outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dequant needs static shapes");

    uint64_t rows = inTy.getShape()[0];
    uint64_t outRows = outTy.getShape()[0];
    Value stride1 = cstI64(rewriter, loc, 1);

    auto in0 = rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    auto out0 = rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    out0->setAttr("row", rewriter.getI64IntegerAttr(4));
    auto in1 = rewriter.create<buckyball::BankMvinOp>(
        loc, rewriter.getI64Type(), input, in0.getBank(),
        cstI64(rewriter, loc, rows), stride1);
    auto out1 = rewriter.create<buckyball::BankDequantOp>(
        loc, rewriter.getI64Type(), in1.getBankOut(), out0.getBank(),
        cstI64(rewriter, loc, rows), op.getScale());
    auto out2 = rewriter.create<buckyball::BankMvoutOp>(
        loc, rewriter.getI64Type(), output, out1.getOutBankOut(),
        cstI64(rewriter, loc, outRows), stride1);

    rewriter.create<buckyball::BankReleaseOp>(loc, in1.getBankOut());
    rewriter.create<buckyball::BankReleaseOp>(loc, out2.getBankOut());
    rewriter.eraseOp(op);
    return success();
  }
};

class LowerBuckyballToBankSSAPass
    : public PassWrapper<LowerBuckyballToBankSSAPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBuckyballToBankSSAPass)
  StringRef getArgument() const final { return "lower-buckyball-to-bank-ssa"; }
  StringRef getDescription() const final {
    return "Lower bb_matmul to explicit bank-SSA ops.";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatMulToBankSSAPattern, TransposeToBankSSAPattern,
                 Im2colToBankSSAPattern, QuantToBankSSAPattern,
                 DequantToBankSSAPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerLowerBuckyballToBankSSAPass() {
  PassRegistration<LowerBuckyballToBankSSAPass>();
}
} // namespace buddy
} // namespace mlir
