//====- LowerLinalgToTile.cpp - Linalg Dialect Lowering Pass -----------===//
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
// This file defines Linalg dialect lowering pass to Tile dialect.
//
//===----------------------------------------------------------------------===//
#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Tile/TileDialect.h"
#include "Tile/TileOps.h"
using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
static std::optional<int64_t> getUniformAttr(DenseIntElementsAttr attr) {
  if (!attr)
    return 1;
  if (attr.empty())
    return std::nullopt;

  int64_t value = (*attr.begin()).getSExtValue();
  for (llvm::APInt element : attr) {
    if (element.getSExtValue() != value)
      return std::nullopt;
  }
  return value;
}

// Gate for pebble convert-tile-to-buckyball im2col path (HWCF filter).
// Large H/W are OK: TileConv2dLowering tiles OH/OW. Stride s>=1 with pad:
//   oh == (h + padLow + padHigh - kh) / s + 1  (exact).
static bool supportsTileConv(MemRefType inType, MemRefType filterType,
                             MemRefType outType, int64_t stride, int64_t padLow,
                             int64_t padHigh) {
  if (inType.getRank() != 4 || filterType.getRank() != 4 ||
      outType.getRank() != 4)
    return false;
  if (!inType.getElementType().isF32() ||
      !filterType.getElementType().isF32() || !outType.getElementType().isF32())
    return false;
  if (!inType.hasStaticShape() || !filterType.hasStaticShape() ||
      !outType.hasStaticShape())
    return false;
  if (stride < 1 || padLow < 0 || padHigh < 0)
    return false;
  constexpr int64_t kMaxPad = 7;
  if (padLow > kMaxPad || padHigh > kMaxPad)
    return false;

  constexpr int64_t kMaxK = 7;
  constexpr int64_t kMaxIter = 34;
  constexpr int64_t kBankLines = 1024;
  constexpr int64_t kLane = 16;

  auto inShape = inType.getShape();
  auto fShape = filterType.getShape();
  auto outShape = outType.getShape();
  int64_t n = inShape[0], h = inShape[1], w = inShape[2], c = inShape[3];
  int64_t kh = fShape[0], kw = fShape[1], fc = fShape[2], oc = fShape[3];
  int64_t oh = outShape[1], ow = outShape[2];
  if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || kh <= 0 || kw <= 0 || fc <= 0 ||
      oc <= 0 || oh <= 0 || ow <= 0)
    return false;
  if (n != outShape[0] || fc != c || outShape[3] != oc)
    return false;
  if (h != w || oh != ow || kh != kw)
    return false;
  if (kh > kMaxK)
    return false;
  int64_t padded = h + padLow + padHigh;
  if (padded < kh)
    return false;
  if ((padded - kh) % stride != 0)
    return false;
  if ((padded - kh) / stride + 1 != oh)
    return false;

  auto cdiv = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
  for (int64_t t = oh; t >= 1; --t) {
    if (oh % t != 0)
      continue;
    int64_t inSize = (t - 1) * stride + kh;
    if (inSize > kMaxIter)
      continue;
    int64_t rows = cdiv(t * t, kLane) * cdiv(kh * kh, kLane) * kLane;
    if (rows <= kBankLines)
      return true;
  }
  return false;
}

static void getConvPads(Operation *op, int64_t &padLow, int64_t &padHigh) {
  padLow = 0;
  padHigh = 0;
  if (auto a = op->getAttrOfType<IntegerAttr>("bb_pad_low"))
    padLow = a.getInt();
  if (auto a = op->getAttrOfType<IntegerAttr>("bb_pad_high"))
    padHigh = a.getInt();
}

class MatmulLowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  explicit MatmulLowering(MLIRContext *context) : OpRewritePattern(context) {}
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulOp matMulOp,
                                PatternRewriter &rewriter) const override {
    auto inputs = matMulOp.getInputs();
    auto ouputs = matMulOp.getOutputs();
    Location loc = matMulOp.getLoc();
    Value input0 = inputs[0];
    Value input1 = inputs[1];
    Value output0 = ouputs[0];
    MemRefType input0Type = dyn_cast<MemRefType>(input0.getType());
    MemRefType input1Type = dyn_cast<MemRefType>(input1.getType());
    MemRefType outputType = dyn_cast<MemRefType>(output0.getType());
    if (!input0Type || !input1Type || !outputType)
      return failure();
    Attribute indexingMaps = matMulOp->getAttr("indexing_maps");
    bool isDefaultMatmul =
        linalg::MatmulOp::isDefaultIndexingMaps(indexingMaps);
    bool isTransposeB =
        linalg::MatmulTransposeBOp::isDefaultIndexingMaps(indexingMaps);
    if (!isDefaultMatmul && !isTransposeB)
      return failure();

    bool needCollapse = false;
    SmallVector<int64_t, 3> aShape;
    SmallVector<int64_t, 3> bShape;
    SmallVector<int64_t, 3> oShape;
    aShape.append(input0Type.getShape().begin(), input0Type.getShape().end());
    bShape.append(input1Type.getShape().begin(), input1Type.getShape().end());
    oShape.append(outputType.getShape().begin(), outputType.getShape().end());
    if (input0Type.getRank() == 3 && input1Type.getRank() == 3 &&
        outputType.getRank() == 3 && aShape[0] == 1 && bShape[0] == 1 &&
        oShape[0] == 1)
      needCollapse = true;
    Value aVal = input0;
    Value bVal = input1;
    Value oVal = output0;
    if (needCollapse) {
      SmallVector<SmallVector<int64_t, 2>, 2> reassoc = {{0, 1}, {2}};
      aVal = memref::CollapseShapeOp::create(rewriter, loc, input0, reassoc);
      bVal = memref::CollapseShapeOp::create(rewriter, loc, input1, reassoc);
      oVal = memref::CollapseShapeOp::create(rewriter, loc, output0, reassoc);
    }

    Value matmulInput1 = bVal;
    if (isTransposeB) {
      auto bValType = dyn_cast<MemRefType>(bVal.getType());
      if (!bValType || bValType.getRank() != 2 || !bValType.hasStaticShape())
        return failure();

      ArrayRef<int64_t> bValShape = bValType.getShape();
      int64_t n = bValShape[0];
      int64_t k = bValShape[1];
      auto transposedType = MemRefType::get({k, n}, bValType.getElementType());
      Value transposed = memref::AllocOp::create(rewriter, loc, transposedType);

      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
      Value nUb = arith::ConstantIndexOp::create(rewriter, loc, n);
      Value kUb = arith::ConstantIndexOp::create(rewriter, loc, k);

      auto nLoop = scf::ForOp::create(rewriter, loc, zero, nUb, one);
      rewriter.setInsertionPointToStart(nLoop.getBody());
      Value nIv = nLoop.getInductionVar();
      auto kLoop = scf::ForOp::create(rewriter, loc, zero, kUb, one);
      rewriter.setInsertionPointToStart(kLoop.getBody());
      Value kIv = kLoop.getInductionVar();
      Value value =
          memref::LoadOp::create(rewriter, loc, bVal, ValueRange{nIv, kIv});
      memref::StoreOp::create(rewriter, loc, value, transposed,
                              ValueRange{kIv, nIv});

      rewriter.setInsertionPointAfter(nLoop);
      matmulInput1 = transposed;
    }

    tile::TileMatMulOp::create(rewriter, loc, aVal, matmulInput1, oVal);
    if (isTransposeB)
      memref::DeallocOp::create(rewriter, loc, matmulInput1);
    rewriter.eraseOp(matMulOp);
    return success();
  }

private:
};

class QuantizedGenericLowering : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto quantized = op->getAttrOfType<BoolAttr>("buckyball.quantized");
    if (!quantized || !quantized.getValue())
      return failure();
    if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
      return op.emitError(
          "quantized generic requires two inputs and one output");

    Value activation = op.getInputs()[0];
    Value weight = op.getInputs()[1];
    Value output = op.getOutputs()[0];
    auto activationType = dyn_cast<MemRefType>(activation.getType());
    auto weightType = dyn_cast<MemRefType>(weight.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());
    if (!activationType || !weightType || !outputType ||
        activationType.getRank() != 2 || weightType.getRank() != 2 ||
        outputType.getRank() != 2 || !activationType.getElementType().isF32() ||
        !weightType.getElementType().isInteger(8) ||
        !outputType.getElementType().isF32())
      return op.emitError(
          "quantized generic requires FP32 x INT8 -> FP32 memrefs");

    auto dwAddr = op->getAttrOfType<IntegerAttr>("dw_addr");
    auto dwBytes = op->getAttrOfType<IntegerAttr>("dw_bytes");
    auto perChannel = op->getAttrOfType<BoolAttr>("per_channel");
    if (!dwAddr || !dwBytes || !perChannel)
      return op.emitError("quantized generic requires RAX Dw metadata");

    auto tileMatmul = tile::TileMatMulOp::create(rewriter, op.getLoc(),
                                                 activation, weight, output);
    tileMatmul->setAttr("dw_addr", dwAddr);
    tileMatmul->setAttr("dw_bytes", dwBytes);
    tileMatmul->setAttr("per_channel", perChannel);
    rewriter.eraseOp(op);
    return success();
  }
};

class BatchMatMulOpLowering : public OpRewritePattern<linalg::BatchMatmulOp> {
public:
  using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::BatchMatmulOp batchMatMulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = batchMatMulOp.getLoc();
    auto inputs = batchMatMulOp.getInputs();
    Value input0 = inputs[0];
    Value input1 = inputs[1];
    Value output = batchMatMulOp.getOutputs()[0];
    MemRefType input0Type = dyn_cast<MemRefType>(input0.getType());
    MemRefType input1Type = dyn_cast<MemRefType>(input1.getType());
    MemRefType outputType = dyn_cast<MemRefType>(output.getType());
    if (!input0Type || !input1Type || !outputType)
      return failure();
    Attribute indexingMaps = batchMatMulOp->getAttr("indexing_maps");
    bool isDefaultBatchMatmul =
        linalg::BatchMatmulOp::isDefaultIndexingMaps(indexingMaps);
    bool isTransposeB =
        linalg::BatchMatmulTransposeBOp::isDefaultIndexingMaps(indexingMaps);
    if (!isDefaultBatchMatmul && !isTransposeB)
      return failure();

    ArrayRef<int64_t> input0Shape = input0Type.getShape();
    ArrayRef<int64_t> input1Shape = input1Type.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    Type elemType = input0Type.getElementType();
    for (unsigned i = 0; i != input0Shape[0]; i++) {
      SmallVector<int64_t> staticOffsets = {i, 0, 0};
      SmallVector<int64_t> staticSizes = {1, input0Shape[1], input0Shape[2]};
      SmallVector<int64_t> staticStrides = {1, 1, 1};
      Value subInput0 = memref::SubViewOp::create(
          rewriter, loc, input0, staticOffsets, staticSizes, staticStrides);
      if (dyn_cast<MemRefType>(subInput0.getType()).getRank() == 3 &&
          dyn_cast<MemRefType>(subInput0.getType()).getShape()[0] == 1) {
        SmallVector<SmallVector<int64_t, 2>, 2> reassoc = {{0, 1}, {2}};
        subInput0 =
            memref::CollapseShapeOp::create(rewriter, loc, subInput0, reassoc);
      }

      staticSizes.assign({1, input1Shape[1], input1Shape[2]});
      Value subInput1 = memref::SubViewOp::create(
          rewriter, loc, input1, staticOffsets, staticSizes, staticStrides);
      if (dyn_cast<MemRefType>(subInput1.getType()).getRank() == 3 &&
          dyn_cast<MemRefType>(subInput1.getType()).getShape()[0] == 1) {
        SmallVector<SmallVector<int64_t, 2>, 2> reassoc = {{0, 1}, {2}};
        subInput1 =
            memref::CollapseShapeOp::create(rewriter, loc, subInput1, reassoc);
      }
      Value matmulInput1 = subInput1;
      if (isTransposeB) {
        auto transposedType =
            MemRefType::get({input1Shape[2], input1Shape[1]}, elemType);
        Value transposed =
            memref::AllocOp::create(rewriter, loc, transposedType);

        Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value nUb =
            arith::ConstantIndexOp::create(rewriter, loc, input1Shape[1]);
        Value kUb =
            arith::ConstantIndexOp::create(rewriter, loc, input1Shape[2]);

        auto nLoop = scf::ForOp::create(rewriter, loc, zero, nUb, one);
        rewriter.setInsertionPointToStart(nLoop.getBody());
        Value nIv = nLoop.getInductionVar();
        auto kLoop = scf::ForOp::create(rewriter, loc, zero, kUb, one);
        rewriter.setInsertionPointToStart(kLoop.getBody());
        Value kIv = kLoop.getInductionVar();
        Value value = memref::LoadOp::create(rewriter, loc, subInput1,
                                             ValueRange{nIv, kIv});
        memref::StoreOp::create(rewriter, loc, value, transposed,
                                ValueRange{kIv, nIv});

        rewriter.setInsertionPointAfter(nLoop);
        matmulInput1 = transposed;
      }

      staticSizes.assign({1, outputShape[1], outputShape[2]});
      Value subOutput = memref::SubViewOp::create(
          rewriter, loc, output, staticOffsets, staticSizes, staticStrides);
      if (dyn_cast<MemRefType>(subOutput.getType()).getRank() == 3 &&
          dyn_cast<MemRefType>(subOutput.getType()).getShape()[0] == 1) {
        SmallVector<SmallVector<int64_t, 2>, 2> reassoc = {{0, 1}, {2}};
        subOutput =
            memref::CollapseShapeOp::create(rewriter, loc, subOutput, reassoc);
      }
      SmallVector<Value> inputs = {subInput0, matmulInput1};
      SmallVector<Value> outputs = {subOutput};
      linalg::MatmulOp::create(rewriter, batchMatMulOp.getLoc(), inputs,
                               outputs);
      if (isTransposeB)
        memref::DeallocOp::create(rewriter, loc, matmulInput1);
    }
    rewriter.eraseOp(batchMatMulOp.getOperation());
    return success();
  }
};

class TransposeOpLowering : public OpRewritePattern<linalg::TransposeOp> {
public:
  explicit TransposeOpLowering(MLIRContext *context)
      : OpRewritePattern<linalg::TransposeOp>(context) {}

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    Value input = transposeOp.getInput();
    Value output = transposeOp.getInit();
    Location loc = transposeOp.getLoc();
    auto inT = dyn_cast<MemRefType>(input.getType());
    auto outT = dyn_cast<MemRefType>(output.getType());
    if (!inT || !outT || !inT.hasStaticShape() || !outT.hasStaticShape() ||
        inT.getElementType() != outT.getElementType())
      return transposeOp.emitOpError("expected static memref operands");

    ArrayRef<int64_t> perm = transposeOp.getPermutation();
    ArrayRef<int64_t> is = inT.getShape();
    ArrayRef<int64_t> os = outT.getShape();
    Type elem = inT.getElementType();
    SmallVector<Value> temps;

    // Copy to identity-layout buffer when needed (collapse/tile require it).
    auto asContig = [&](Value v) -> Value {
      auto t = cast<MemRefType>(v.getType());
      if (t.getLayout().isIdentity())
        return v;
      Value b = memref::AllocOp::create(
          rewriter, loc, MemRefType::get(t.getShape(), t.getElementType()));
      memref::CopyOp::create(rewriter, loc, v, b);
      temps.push_back(b);
      return b;
    };
    auto dstContig = [&]() -> Value {
      if (outT.getLayout().isIdentity())
        return output;
      Value b =
          memref::AllocOp::create(rewriter, loc, MemRefType::get(os, elem));
      temps.push_back(b);
      return b;
    };
    auto collapse = [&](Value src, ArrayRef<int64_t> shape2,
                        ArrayRef<ReassociationIndices> reassoc) -> Value {
      return memref::CollapseShapeOp::create(
          rewriter, loc, MemRefType::get(shape2, elem), src, reassoc);
    };
    auto finish = [&](Value dst) {
      if (dst != output)
        memref::CopyOp::create(rewriter, loc, dst, output);
      for (Value t : llvm::reverse(temps))
        memref::DeallocOp::create(rewriter, loc, t);
      rewriter.eraseOp(transposeOp);
      return success();
    };

    // Copy subview into a fresh contiguous buffer, then collapse to 2D.
    auto slice2D = [&](Value src4, ArrayRef<int64_t> off, ArrayRef<int64_t> sz4,
                       ArrayRef<int64_t> shape2,
                       ArrayRef<ReassociationIndices> reassoc) -> Value {
      Value sub = memref::SubViewOp::create(rewriter, loc, src4, off, sz4,
                                            SmallVector<int64_t>(4, 1));
      Value buf =
          memref::AllocOp::create(rewriter, loc, MemRefType::get(sz4, elem));
      memref::CopyOp::create(rewriter, loc, sub, buf);
      Value flat = collapse(buf, shape2, reassoc);
      temps.push_back(buf);
      return flat;
    };

    if (inT.getRank() == 2) {
      if (perm != ArrayRef<int64_t>({1, 0}))
        return transposeOp.emitOpError("rank-2 requires perm [1,0]");
      Value dst = dstContig();
      tile::TileTransposeOp::create(rewriter, loc, asContig(input), dst);
      return finish(dst);
    }

    if (inT.getRank() == 3) {
      if (perm != ArrayRef<int64_t>({0, 2, 1}) || is[0] != 1)
        return transposeOp.emitOpError("rank-3 only N=1 perm [0,2,1]");
      Value src = asContig(input), dst = dstContig();
      tile::TileTransposeOp::create(
          rewriter, loc, collapse(src, {is[1], is[2]}, {{0, 1}, {2}}),
          collapse(dst, {os[1], os[2]}, {{0, 1}, {2}}));
      return finish(dst);
    }

    if (inT.getRank() != 4)
      return transposeOp.emitOpError("unsupported transpose rank");

    // 1x1 OIHW->OHWI is layout-identical after collapse to [O,I].
    if (perm == ArrayRef<int64_t>({0, 2, 3, 1}) && is[2] == 1 && is[3] == 1) {
      Value src = asContig(input), dst = dstContig();
      memref::CopyOp::create(rewriter, loc,
                             collapse(src, {is[0], is[1]}, {{0}, {1, 2, 3}}),
                             collapse(dst, {os[0], os[3]}, {{0}, {1, 2, 3}}));
      return finish(dst);
    }

    // N=1 NCHW->NHWC: [1,C,H,W] -> [1,H,W,C]
    if (perm == ArrayRef<int64_t>({0, 2, 3, 1}) && is[0] == 1) {
      int64_t c = is[1], hw = is[2] * is[3];
      Value src = asContig(input), dst = dstContig();
      tile::TileTransposeOp::create(rewriter, loc,
                                    collapse(src, {c, hw}, {{0, 1}, {2, 3}}),
                                    collapse(dst, {hw, c}, {{0, 1, 2}, {3}}));
      return finish(dst);
    }

    // Weight OIHW->OHWI: per-O 2D transpose of I x (H*W)
    if (perm == ArrayRef<int64_t>({0, 2, 3, 1})) {
      int64_t o = is[0], i = is[1], h = is[2], w = is[3], hw = h * w;
      Value src = asContig(input), dst = dstContig();
      for (int64_t oi = 0; oi < o; ++oi) {
        SmallVector<int64_t> off = {oi, 0, 0, 0};
        Value in2 = slice2D(src, off, {1, i, h, w}, {i, hw}, {{0, 1}, {2, 3}});
        Value outBuf = memref::AllocOp::create(
            rewriter, loc, MemRefType::get({1, h, w, i}, elem));
        Value out2 = collapse(outBuf, {hw, i}, {{0, 1, 2}, {3}});
        tile::TileTransposeOp::create(rewriter, loc, in2, out2);
        memref::CopyOp::create(
            rewriter, loc, outBuf,
            memref::SubViewOp::create(rewriter, loc, dst, off, {1, h, w, i},
                                      SmallVector<int64_t>(4, 1)));
        memref::DeallocOp::create(rewriter, loc, outBuf);
      }
      return finish(dst);
    }

    // N=1 NHWC->NCHW / attention: [1,A,B,C] -> [1,C,A,B]
    if (perm == ArrayRef<int64_t>({0, 3, 1, 2}) && is[0] == 1) {
      int64_t ab = is[1] * is[2], c = is[3];
      Value src = asContig(input), dst = dstContig();
      tile::TileTransposeOp::create(rewriter, loc,
                                    collapse(src, {ab, c}, {{0, 1, 2}, {3}}),
                                    collapse(dst, {c, ab}, {{0, 1}, {2, 3}}));
      return finish(dst);
    }

    // [N,A,B,C] -> [N,C,B,A] (e.g. NHWC->NCWH). N=1: one [AB,C]->[C,AB]
    // then per-channel [A,B]->[B,A]. Else loop the cheaper of A/B.
    if (perm == ArrayRef<int64_t>({0, 3, 2, 1})) {
      int64_t n = is[0], a = is[1], b = is[2], c = is[3];
      Value src = asContig(input), dst = dstContig();
      if (n == 1) {
        Value tmp = memref::AllocOp::create(
            rewriter, loc, MemRefType::get({1, c, a, b}, elem));
        temps.push_back(tmp);
        tile::TileTransposeOp::create(
            rewriter, loc, collapse(src, {a * b, c}, {{0, 1, 2}, {3}}),
            collapse(tmp, {c, a * b}, {{0, 1}, {2, 3}}));
        for (int64_t ci = 0; ci < c; ++ci) {
          SmallVector<int64_t> off = {0, ci, 0, 0};
          Value in2 = slice2D(tmp, off, {1, 1, a, b}, {a, b}, {{0, 1, 2}, {3}});
          Value outBuf = memref::AllocOp::create(
              rewriter, loc, MemRefType::get({1, 1, b, a}, elem));
          Value out2 = collapse(outBuf, {b, a}, {{0, 1, 2}, {3}});
          tile::TileTransposeOp::create(rewriter, loc, in2, out2);
          memref::CopyOp::create(
              rewriter, loc, outBuf,
              memref::SubViewOp::create(rewriter, loc, dst, off, {1, 1, b, a},
                                        SmallVector<int64_t>(4, 1)));
          memref::DeallocOp::create(rewriter, loc, outBuf);
        }
        return finish(dst);
      }
      bool loopB = b <= a;
      for (int64_t ni = 0; ni < n; ++ni) {
        if (loopB) {
          for (int64_t bi = 0; bi < b; ++bi) {
            SmallVector<int64_t> off = {ni, 0, bi, 0};
            Value in2 =
                slice2D(src, off, {1, a, 1, c}, {a, c}, {{0, 1}, {2, 3}});
            Value outBuf = memref::AllocOp::create(
                rewriter, loc, MemRefType::get({1, c, 1, a}, elem));
            Value out2 = collapse(outBuf, {c, a}, {{0, 1}, {2, 3}});
            tile::TileTransposeOp::create(rewriter, loc, in2, out2);
            memref::CopyOp::create(
                rewriter, loc, outBuf,
                memref::SubViewOp::create(rewriter, loc, dst, off, {1, c, 1, a},
                                          SmallVector<int64_t>(4, 1)));
            memref::DeallocOp::create(rewriter, loc, outBuf);
          }
        } else {
          for (int64_t ai = 0; ai < a; ++ai) {
            SmallVector<int64_t> inOff = {ni, ai, 0, 0};
            SmallVector<int64_t> outOff = {ni, 0, 0, ai};
            Value in2 =
                slice2D(src, inOff, {1, 1, b, c}, {b, c}, {{0, 1, 2}, {3}});
            Value outBuf = memref::AllocOp::create(
                rewriter, loc, MemRefType::get({1, c, b, 1}, elem));
            Value out2 = collapse(outBuf, {c, b}, {{0, 1}, {2, 3}});
            tile::TileTransposeOp::create(rewriter, loc, in2, out2);
            memref::CopyOp::create(rewriter, loc, outBuf,
                                   memref::SubViewOp::create(
                                       rewriter, loc, dst, outOff, {1, c, b, 1},
                                       SmallVector<int64_t>(4, 1)));
            memref::DeallocOp::create(rewriter, loc, outBuf);
          }
        }
      }
      return finish(dst);
    }

    // [N,A,B,C] -> [N,B,A,C]
    if (perm == ArrayRef<int64_t>({0, 2, 1, 3})) {
      int64_t n = is[0], a = is[1], b = is[2], c = is[3];
      Value src = asContig(input), dst = dstContig();
      for (int64_t ni = 0; ni < n; ++ni) {
        for (int64_t ci = 0; ci < c; ++ci) {
          SmallVector<int64_t> off = {ni, 0, 0, ci};
          Value in2 = slice2D(src, off, {1, a, b, 1}, {a, b}, {{0, 1}, {2, 3}});
          Value outBuf = memref::AllocOp::create(
              rewriter, loc, MemRefType::get({1, b, a, 1}, elem));
          Value out2 = collapse(outBuf, {b, a}, {{0, 1}, {2, 3}});
          tile::TileTransposeOp::create(rewriter, loc, in2, out2);
          memref::CopyOp::create(
              rewriter, loc, outBuf,
              memref::SubViewOp::create(rewriter, loc, dst, off, {1, b, a, 1},
                                        SmallVector<int64_t>(4, 1)));
          memref::DeallocOp::create(rewriter, loc, outBuf);
        }
      }
      return finish(dst);
    }

    // [N,H,A,B] -> [N,H,B,A]: per-(n,h) 2D transpose
    if (perm == ArrayRef<int64_t>({0, 1, 3, 2})) {
      int64_t n = is[0], h = is[1], a = is[2], b = is[3];
      Value src = asContig(input), dst = dstContig();
      for (int64_t ni = 0; ni < n; ++ni) {
        for (int64_t hi = 0; hi < h; ++hi) {
          SmallVector<int64_t> off = {ni, hi, 0, 0};
          Value in2 = slice2D(src, off, {1, 1, a, b}, {a, b}, {{0, 1, 2}, {3}});
          Value outBuf = memref::AllocOp::create(
              rewriter, loc, MemRefType::get({1, 1, b, a}, elem));
          Value out2 = collapse(outBuf, {b, a}, {{0, 1, 2}, {3}});
          tile::TileTransposeOp::create(rewriter, loc, in2, out2);
          memref::CopyOp::create(
              rewriter, loc, outBuf,
              memref::SubViewOp::create(rewriter, loc, dst, off, {1, 1, b, a},
                                        SmallVector<int64_t>(4, 1)));
          memref::DeallocOp::create(rewriter, loc, outBuf);
        }
      }
      return finish(dst);
    }

    // Depthwise OIHW->HWIO: [C,1,k,k] -> [k,k,C,1]
    if (perm == ArrayRef<int64_t>({2, 3, 0, 1}) && is[1] == 1) {
      int64_t c = is[0], kk = is[2] * is[3];
      Value src = asContig(input), dst = dstContig();
      tile::TileTransposeOp::create(rewriter, loc,
                                    collapse(src, {c, kk}, {{0, 1}, {2, 3}}),
                                    collapse(dst, {kk, c}, {{0, 1}, {2, 3}}));
      return finish(dst);
    }

    return transposeOp.emitOpError(
        "unsupported transpose perm/shape for tile lowering");
  }
};

class Conv2dNhwcHwcfLowering
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
public:
  using OpRewritePattern<linalg::Conv2DNhwcHwcfOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    auto inputs = convOp.getInputs();
    auto outputs = convOp.getOutputs();
    if (inputs.size() != 2 || outputs.size() != 1)
      return failure();
    auto stride = getUniformAttr(convOp.getStrides());
    auto dilation = getUniformAttr(convOp.getDilations());
    if (!stride || !dilation || *stride < 1 || *dilation != 1)
      return failure();

    Value input = inputs[0];
    Value filter = inputs[1];
    Value output = outputs[0];
    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto filterType = dyn_cast<MemRefType>(filter.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());
    if (!inputType || !filterType || !outputType)
      return failure();
    int64_t padLow = 0, padHigh = 0;
    getConvPads(convOp, padLow, padHigh);
    if (!supportsTileConv(inputType, filterType, outputType, *stride, padLow,
                          padHigh))
      return failure();
    rewriter.replaceOpWithNewOp<tile::TileConv2dOp>(
        convOp, input, filter, output, rewriter.getI64IntegerAttr(padLow),
        rewriter.getI64IntegerAttr(padHigh));
    return success();
  }
};

class Conv2dNhwcFhwcLowering
    : public OpRewritePattern<linalg::Conv2DNhwcFhwcOp> {
public:
  using OpRewritePattern<linalg::Conv2DNhwcFhwcOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::Conv2DNhwcFhwcOp convOp,
                                PatternRewriter &rewriter) const override {
    auto inputs = convOp.getInputs();
    auto outputs = convOp.getOutputs();
    if (inputs.size() != 2 || outputs.size() != 1)
      return failure();

    Value input = inputs[0];
    Value filter = inputs[1];
    Value output = outputs[0];
    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto filterType = dyn_cast<MemRefType>(filter.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());
    if (!inputType || !filterType || !outputType)
      return failure();
    if (inputType.getRank() != 4 || filterType.getRank() != 4 ||
        outputType.getRank() != 4)
      return failure();

    auto stride = getUniformAttr(convOp.getStrides());
    auto dilation = getUniformAttr(convOp.getDilations());
    if (!stride || !dilation || *stride < 1 || *dilation != 1)
      return failure();

    Location loc = convOp.getLoc();
    ArrayRef<int64_t> filterShape = filterType.getShape();
    int64_t oc = filterShape[0];
    int64_t kh = filterShape[1];
    int64_t kw = filterShape[2];
    int64_t c = filterShape[3];

    auto hwcfType =
        MemRefType::get({kh, kw, c, oc}, filterType.getElementType());
    int64_t padLow = 0, padHigh = 0;
    getConvPads(convOp, padLow, padHigh);
    if (!supportsTileConv(inputType, hwcfType, outputType, *stride, padLow,
                          padHigh))
      return failure();
    Value hwcf = memref::AllocOp::create(rewriter, loc, hwcfType);

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value ocUb = arith::ConstantIndexOp::create(rewriter, loc, oc);
    Value khUb = arith::ConstantIndexOp::create(rewriter, loc, kh);
    Value kwUb = arith::ConstantIndexOp::create(rewriter, loc, kw);
    Value cUb = arith::ConstantIndexOp::create(rewriter, loc, c);

    auto ocLoop = scf::ForOp::create(rewriter, loc, zero, ocUb, one);
    rewriter.setInsertionPointToStart(ocLoop.getBody());
    Value ocIv = ocLoop.getInductionVar();

    auto khLoop = scf::ForOp::create(rewriter, loc, zero, khUb, one);
    rewriter.setInsertionPointToStart(khLoop.getBody());
    Value khIv = khLoop.getInductionVar();

    auto kwLoop = scf::ForOp::create(rewriter, loc, zero, kwUb, one);
    rewriter.setInsertionPointToStart(kwLoop.getBody());
    Value kwIv = kwLoop.getInductionVar();

    auto cLoop = scf::ForOp::create(rewriter, loc, zero, cUb, one);
    rewriter.setInsertionPointToStart(cLoop.getBody());
    Value cIv = cLoop.getInductionVar();

    Value value = memref::LoadOp::create(rewriter, loc, filter,
                                         ValueRange{ocIv, khIv, kwIv, cIv});
    memref::StoreOp::create(rewriter, loc, value, hwcf,
                            ValueRange{khIv, kwIv, cIv, ocIv});

    rewriter.setInsertionPointAfter(ocLoop);
    tile::TileConv2dOp::create(rewriter, loc, input, hwcf, output,
                               rewriter.getI64IntegerAttr(padLow),
                               rewriter.getI64IntegerAttr(padHigh));
    memref::DeallocOp::create(rewriter, loc, hwcf);
    rewriter.eraseOp(convOp);
    return success();
  }
};

class Conv2dNchwFchwLowering
    : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
public:
  using OpRewritePattern<linalg::Conv2DNchwFchwOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter &rewriter) const override {
    auto inputs = convOp.getInputs();
    auto outputs = convOp.getOutputs();
    if (inputs.size() != 2 || outputs.size() != 1)
      return failure();

    Value input = inputs[0];
    Value filter = inputs[1];
    Value output = outputs[0];
    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto filterType = dyn_cast<MemRefType>(filter.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());
    if (!inputType || !filterType || !outputType)
      return failure();
    if (inputType.getRank() != 4 || filterType.getRank() != 4 ||
        outputType.getRank() != 4)
      return failure();

    auto stride = getUniformAttr(convOp.getStrides());
    auto dilation = getUniformAttr(convOp.getDilations());
    if (!stride || !dilation || *stride < 1 || *dilation != 1)
      return failure();

    Location loc = convOp.getLoc();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> filterShape = filterType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    int64_t n = inputShape[0];
    int64_t c = inputShape[1];
    int64_t h = inputShape[2];
    int64_t w = inputShape[3];
    int64_t f = filterShape[0];
    int64_t kh = filterShape[2];
    int64_t kw = filterShape[3];
    int64_t oh = outputShape[2];
    int64_t ow = outputShape[3];

    auto nhwcType = MemRefType::get({n, h, w, c}, inputType.getElementType());
    auto hwcfType =
        MemRefType::get({kh, kw, c, f}, filterType.getElementType());
    auto outNhwcType =
        MemRefType::get({n, oh, ow, f}, outputType.getElementType());
    int64_t padLow = 0, padHigh = 0;
    getConvPads(convOp, padLow, padHigh);
    if (!supportsTileConv(nhwcType, hwcfType, outNhwcType, *stride, padLow,
                          padHigh))
      return failure();
    Value nhwc = memref::AllocOp::create(rewriter, loc, nhwcType);
    Value hwcf = memref::AllocOp::create(rewriter, loc, hwcfType);
    Value outNhwc = memref::AllocOp::create(rewriter, loc, outNhwcType);

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value nUb = arith::ConstantIndexOp::create(rewriter, loc, n);
    Value cUb = arith::ConstantIndexOp::create(rewriter, loc, c);
    Value hUb = arith::ConstantIndexOp::create(rewriter, loc, h);
    Value wUb = arith::ConstantIndexOp::create(rewriter, loc, w);
    Value fUb = arith::ConstantIndexOp::create(rewriter, loc, f);
    Value khUb = arith::ConstantIndexOp::create(rewriter, loc, kh);
    Value kwUb = arith::ConstantIndexOp::create(rewriter, loc, kw);
    Value ohUb = arith::ConstantIndexOp::create(rewriter, loc, oh);
    Value owUb = arith::ConstantIndexOp::create(rewriter, loc, ow);

    auto nLoop = scf::ForOp::create(rewriter, loc, zero, nUb, one);
    rewriter.setInsertionPointToStart(nLoop.getBody());
    Value nIv = nLoop.getInductionVar();
    auto cLoop = scf::ForOp::create(rewriter, loc, zero, cUb, one);
    rewriter.setInsertionPointToStart(cLoop.getBody());
    Value cIv = cLoop.getInductionVar();
    auto hLoop = scf::ForOp::create(rewriter, loc, zero, hUb, one);
    rewriter.setInsertionPointToStart(hLoop.getBody());
    Value hIv = hLoop.getInductionVar();
    auto wLoop = scf::ForOp::create(rewriter, loc, zero, wUb, one);
    rewriter.setInsertionPointToStart(wLoop.getBody());
    Value wIv = wLoop.getInductionVar();
    Value inputValue = memref::LoadOp::create(rewriter, loc, input,
                                              ValueRange{nIv, cIv, hIv, wIv});
    memref::StoreOp::create(rewriter, loc, inputValue, nhwc,
                            ValueRange{nIv, hIv, wIv, cIv});

    rewriter.setInsertionPointAfter(nLoop);
    auto fLoop = scf::ForOp::create(rewriter, loc, zero, fUb, one);
    rewriter.setInsertionPointToStart(fLoop.getBody());
    Value fIv = fLoop.getInductionVar();
    auto fcLoop = scf::ForOp::create(rewriter, loc, zero, cUb, one);
    rewriter.setInsertionPointToStart(fcLoop.getBody());
    Value fcIv = fcLoop.getInductionVar();
    auto khLoop = scf::ForOp::create(rewriter, loc, zero, khUb, one);
    rewriter.setInsertionPointToStart(khLoop.getBody());
    Value khIv = khLoop.getInductionVar();
    auto kwLoop = scf::ForOp::create(rewriter, loc, zero, kwUb, one);
    rewriter.setInsertionPointToStart(kwLoop.getBody());
    Value kwIv = kwLoop.getInductionVar();
    Value filterValue = memref::LoadOp::create(
        rewriter, loc, filter, ValueRange{fIv, fcIv, khIv, kwIv});
    memref::StoreOp::create(rewriter, loc, filterValue, hwcf,
                            ValueRange{khIv, kwIv, fcIv, fIv});

    rewriter.setInsertionPointAfter(fLoop);
    tile::TileConv2dOp::create(rewriter, loc, nhwc, hwcf, outNhwc,
                               rewriter.getI64IntegerAttr(padLow),
                               rewriter.getI64IntegerAttr(padHigh));

    auto onLoop = scf::ForOp::create(rewriter, loc, zero, nUb, one);
    rewriter.setInsertionPointToStart(onLoop.getBody());
    Value onIv = onLoop.getInductionVar();
    auto ofLoop = scf::ForOp::create(rewriter, loc, zero, fUb, one);
    rewriter.setInsertionPointToStart(ofLoop.getBody());
    Value ofIv = ofLoop.getInductionVar();
    auto ohLoop = scf::ForOp::create(rewriter, loc, zero, ohUb, one);
    rewriter.setInsertionPointToStart(ohLoop.getBody());
    Value ohIv = ohLoop.getInductionVar();
    auto owLoop = scf::ForOp::create(rewriter, loc, zero, owUb, one);
    rewriter.setInsertionPointToStart(owLoop.getBody());
    Value owIv = owLoop.getInductionVar();
    Value outputValue = memref::LoadOp::create(
        rewriter, loc, outNhwc, ValueRange{onIv, ohIv, owIv, ofIv});
    memref::StoreOp::create(rewriter, loc, outputValue, output,
                            ValueRange{onIv, ofIv, ohIv, owIv});

    rewriter.setInsertionPointAfter(onLoop);
    memref::DeallocOp::create(rewriter, loc, nhwc);
    memref::DeallocOp::create(rewriter, loc, hwcf);
    memref::DeallocOp::create(rewriter, loc, outNhwc);
    rewriter.eraseOp(convOp);
    return success();
  }
};

static bool supportsTileDepthwise(MemRefType inType, MemRefType filterType,
                                  MemRefType outType, int64_t stride,
                                  int64_t padLow, int64_t padHigh) {
  if (inType.getRank() != 4 || filterType.getRank() != 4 ||
      outType.getRank() != 4)
    return false;
  if (!inType.getElementType().isF32() ||
      !filterType.getElementType().isF32() || !outType.getElementType().isF32())
    return false;
  if (!inType.hasStaticShape() || !filterType.hasStaticShape() ||
      !outType.hasStaticShape())
    return false;
  if (stride < 1 || padLow < 0 || padHigh < 0)
    return false;
  constexpr int64_t kMaxPad = 7;
  if (padLow > kMaxPad || padHigh > kMaxPad)
    return false;

  constexpr int64_t kMaxK = 7;
  constexpr int64_t kMaxIter = 34;
  constexpr int64_t kBankLines = 1024;
  constexpr int64_t kLane = 16;

  auto inShape = inType.getShape();
  auto fShape = filterType.getShape();
  auto outShape = outType.getShape();
  int64_t n = inShape[0], h = inShape[1], w = inShape[2], c = inShape[3];
  int64_t kh = fShape[0], kw = fShape[1], fc = fShape[2], mult = fShape[3];
  int64_t oh = outShape[1], ow = outShape[2], oc = outShape[3];
  if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || kh <= 0 || kw <= 0 || fc <= 0 ||
      oh <= 0 || ow <= 0)
    return false;
  if (n != outShape[0] || fc != c || oc != c || mult != 1)
    return false;
  if (h != w || oh != ow || kh != kw)
    return false;
  if (kh > kMaxK)
    return false;
  int64_t padded = h + padLow + padHigh;
  if (padded < kh)
    return false;
  if ((padded - kh) % stride != 0)
    return false;
  if ((padded - kh) / stride + 1 != oh)
    return false;

  auto cdiv = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
  for (int64_t t = oh; t >= 1; --t) {
    if (oh % t != 0)
      continue;
    int64_t inSize = (t - 1) * stride + kh;
    if (inSize > kMaxIter)
      continue;
    int64_t rows = cdiv(t * t, kLane) * cdiv(kh * kh, kLane) * kLane;
    if (rows <= kBankLines)
      return true;
  }
  return false;
}

class DepthwiseConv2dNhwcHwcmLowering
    : public OpRewritePattern<linalg::DepthwiseConv2DNhwcHwcmOp> {
public:
  using OpRewritePattern<linalg::DepthwiseConv2DNhwcHwcmOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::DepthwiseConv2DNhwcHwcmOp convOp,
                                PatternRewriter &rewriter) const override {
    auto inputs = convOp.getInputs();
    auto outputs = convOp.getOutputs();
    if (inputs.size() != 2 || outputs.size() != 1)
      return failure();
    auto stride = getUniformAttr(convOp.getStrides());
    auto dilation = getUniformAttr(convOp.getDilations());
    if (!stride || !dilation || *stride < 1 || *dilation != 1)
      return failure();

    Value input = inputs[0];
    Value filter = inputs[1];
    Value output = outputs[0];
    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto filterType = dyn_cast<MemRefType>(filter.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());
    if (!inputType || !filterType || !outputType)
      return failure();
    if (outputType.getRank() != 5 || !outputType.hasStaticShape() ||
        outputType.getShape()[4] != 1)
      return failure();

    Location loc = convOp.getLoc();
    ArrayRef<int64_t> os = outputType.getShape();
    auto out4Type = MemRefType::get({os[0], os[1], os[2], os[3]},
                                    outputType.getElementType());
    SmallVector<ReassociationIndices, 4> reassoc = {{0}, {1}, {2}, {3, 4}};
    Value out4 = memref::CollapseShapeOp::create(rewriter, loc, out4Type,
                                                 output, reassoc);

    int64_t padLow = 0, padHigh = 0;
    getConvPads(convOp, padLow, padHigh);
    if (!supportsTileDepthwise(inputType, filterType, out4Type, *stride, padLow,
                               padHigh))
      return failure();

    tile::TileDepthwiseConv2dOp::create(rewriter, loc, input, filter, out4,
                                        rewriter.getI64IntegerAttr(padLow),
                                        rewriter.getI64IntegerAttr(padHigh));
    rewriter.eraseOp(convOp);
    return success();
  }
};

} // namespace

void populateLowerLinalgToTileConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<QuantizedGenericLowering>(patterns.getContext());
  patterns.add<MatmulLowering>(patterns.getContext());
  patterns.add<BatchMatMulOpLowering>(patterns.getContext());
  patterns.add<TransposeOpLowering>(patterns.getContext());
  patterns.add<Conv2dNhwcHwcfLowering>(patterns.getContext());
  patterns.add<Conv2dNhwcFhwcLowering>(patterns.getContext());
  patterns.add<Conv2dNchwFchwLowering>(patterns.getContext());
  patterns.add<DepthwiseConv2dNhwcHwcmLowering>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// FusePadIntoConv
//===----------------------------------------------------------------------===//

namespace {

static bool isZeroPad(tensor::PadOp pad) {
  if (!pad.getConstantPaddingValue())
    return false;
  auto cst = pad.getConstantPaddingValue().getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return false;
  auto attr = dyn_cast<FloatAttr>(cst.getValue());
  return attr && attr.getValue().isZero();
}

static LogicalResult tryFusePad(Operation *conv, PatternRewriter &rewriter) {
  if (conv->getNumOperands() < 2)
    return failure();
  Value in = conv->getOperand(0);
  auto pad = in.getDefiningOp<tensor::PadOp>();
  if (!pad || !isZeroPad(pad) || !pad.getResult().hasOneUse())
    return failure();

  RankedTensorType srcTy =
      dyn_cast<RankedTensorType>(pad.getSource().getType());
  RankedTensorType dstTy =
      dyn_cast<RankedTensorType>(pad.getResult().getType());
  if (!srcTy || !dstTy || srcTy.getRank() != 4 || !srcTy.hasStaticShape() ||
      !dstTy.hasStaticShape())
    return failure();

  SmallVector<OpFoldResult> low = pad.getMixedLowPad();
  SmallVector<OpFoldResult> high = pad.getMixedHighPad();
  auto asConst = [](OpFoldResult ofr) -> std::optional<int64_t> {
    if (auto attr = dyn_cast<Attribute>(ofr))
      if (auto ia = dyn_cast<IntegerAttr>(attr))
        return ia.getInt();
    return std::nullopt;
  };
  int64_t pads[4][2];
  for (int i = 0; i < 4; ++i) {
    auto lo = asConst(low[i]);
    auto hi = asConst(high[i]);
    if (!lo || !hi)
      return failure();
    pads[i][0] = *lo;
    pads[i][1] = *hi;
  }
  if (pads[0][0] || pads[0][1] || pads[3][0] || pads[3][1])
    return failure();
  if (pads[1][0] != pads[2][0] || pads[1][1] != pads[2][1])
    return failure();
  int64_t padLow = pads[1][0];
  int64_t padHigh = pads[1][1];
  if (padLow < 0 || padHigh < 0 || padLow > 7 || padHigh > 7)
    return failure();

  conv->setOperand(0, pad.getSource());
  conv->setAttr("bb_pad_low", rewriter.getI64IntegerAttr(padLow));
  conv->setAttr("bb_pad_high", rewriter.getI64IntegerAttr(padHigh));
  rewriter.eraseOp(pad);
  return success();
}

class FusePadConvNhwcFhwc : public OpRewritePattern<linalg::Conv2DNhwcFhwcOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::Conv2DNhwcFhwcOp op,
                                PatternRewriter &rewriter) const override {
    return tryFusePad(op, rewriter);
  }
};

class FusePadConvNhwcHwcf : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp op,
                                PatternRewriter &rewriter) const override {
    return tryFusePad(op, rewriter);
  }
};

class FusePadDepthwiseNhwcHwcm
    : public OpRewritePattern<linalg::DepthwiseConv2DNhwcHwcmOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::DepthwiseConv2DNhwcHwcmOp op,
                                PatternRewriter &rewriter) const override {
    return tryFusePad(op, rewriter);
  }
};

class FusePadIntoConvPass
    : public PassWrapper<FusePadIntoConvPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusePadIntoConvPass);
  StringRef getArgument() const final { return "fuse-pad-into-conv"; }
  StringRef getDescription() const final {
    return "Fuse tensor.pad into linalg conv/depthwise as bb_pad_* attrs";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FusePadConvNhwcFhwc, FusePadConvNhwcHwcf,
                 FusePadDepthwiseNhwcHwcm>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                    arith::ArithDialect>();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// LowerLinalgToTile
//===----------------------------------------------------------------------===//

namespace {
class LowerLinalgToTilePass
    : public PassWrapper<LowerLinalgToTilePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToTilePass);
  LowerLinalgToTilePass() = default;
  LowerLinalgToTilePass(const LowerLinalgToTilePass &) {}
  StringRef getArgument() const final { return "convert-linalg-to-tile"; }
  StringRef getDescription() const final {
    return "convert linalg dialect to tile dialect";
  }
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<tile::TileDialect, func::FuncDialect, memref::MemRefDialect,
                linalg::LinalgDialect, arith::ArithDialect, scf::SCFDialect>();
  }
};
} // namespace

void LowerLinalgToTilePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  ConversionTarget target(*context);
  target.addLegalDialect<memref::MemRefDialect, tile::TileDialect,
                         arith::ArithDialect, scf::SCFDialect>();
  target.addLegalOp<linalg::FillOp, linalg::YieldOp>();
  target.addIllegalOp<linalg::TransposeOp>();
  RewritePatternSet patterns(context);
  populateLowerLinalgToTileConversionPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerLinalgToTilePass() {
  PassRegistration<LowerLinalgToTilePass>();
  PassRegistration<FusePadIntoConvPass>();
}
} // namespace buddy
} // namespace mlir
