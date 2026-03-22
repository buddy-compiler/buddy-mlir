//====- LowerTileToBuckyball.cpp - Tile to Buckyball Lowering Pass -------===//
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
// This file defines the pass to lower Tile dialect to Buckyball dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
#include "Tile/TileDialect.h"
#include "Tile/TileOps.h"

using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Helper: ceil division
//===----------------------------------------------------------------------===//

static size_t ceilDiv(size_t a, size_t b) { return (a + b - 1) / b; }

//===----------------------------------------------------------------------===//
// Tile Matmul Lowering Pattern
//===----------------------------------------------------------------------===//

namespace {

class TileMatMulLowering : public OpRewritePattern<tile::TileMatMulOp> {
  // Compute bank rows needed: A occupies mTileLen*kTileLen, B occupies kTileLen*nTileLen
  size_t computeBankRows(size_t mTileLen, size_t nTileLen,
                         size_t kTileLen) const {
    return mTileLen * kTileLen + kTileLen * nTileLen;
  }

public:
  explicit TileMatMulLowering(MLIRContext *context, int64_t lane, int64_t warp,
                              int64_t bankDepth, int64_t /*bankNum*/)
      : OpRewritePattern(context), lane(lane), warp(warp),
        bankDepth(bankDepth) {}

  LogicalResult matchAndRewrite(tile::TileMatMulOp tileMatMulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = tileMatMulOp.getLoc();

    Value aMemArray = tileMatMulOp.getAMemArray();
    Value bMemArray = tileMatMulOp.getBMemArray();
    Value cMemArray = tileMatMulOp.getCMemArray();

    auto aType = cast<MemRefType>(aMemArray.getType());
    auto bType = cast<MemRefType>(bMemArray.getType());

    // A[M][K], B[K][N], C[M][N]
    auto aShape = aType.getShape();
    auto bShape = bType.getShape();
    size_t M = aShape[aShape.size() - 2];
    size_t K = aShape[aShape.size() - 1];
    size_t N = bShape[bShape.size() - 1];

    // Block counts per axis: each unit is lane×lane×warp elements (see mTileSize below).
    const size_t mMeta = lane;
    const size_t nMeta = lane;
    const size_t kMeta = warp;

    // Pad dimensions to multiples of meta lengths
    const size_t mPad = ceilDiv(M, mMeta) * mMeta;
    const size_t nPad = ceilDiv(N, nMeta) * nMeta;
    const size_t kPad = ceilDiv(K, kMeta) * kMeta;

    // Compute tile lengths to maximize utilization within bank capacity
    size_t mTileLen = 1, nTileLen = 1, kTileLen = 1;

    // Extend N dimension first
    while ((nTileLen + 1) * nMeta <= nPad &&
           computeBankRows(mTileLen, nTileLen + 1, kTileLen) <= (size_t)bankDepth)
      nTileLen++;

    // Then extend M dimension
    while ((mTileLen + 1) * mMeta <= mPad &&
           computeBankRows(mTileLen + 1, nTileLen, kTileLen) <= (size_t)bankDepth)
      mTileLen++;

    // Tile sizes and counts
    const size_t mTileSize = mTileLen * mMeta;
    const size_t nTileSize = nTileLen * nMeta;
    const size_t kTileSize = kTileLen * kMeta;
    const size_t mTileNum = ceilDiv(mPad, mTileSize);
    const size_t nTileNum = ceilDiv(nPad, nTileSize);
    const size_t kTileNum = ceilDiv(kPad, kTileSize);

    // Generate tiled computation
    for (size_t k0 = 0; k0 < kTileNum; k0++) {
      for (size_t m0 = 0; m0 < mTileNum; m0++) {
        for (size_t n0 = 0; n0 < nTileNum; n0++) {
          size_t mStart = m0 * mTileSize, kStart = k0 * kTileSize,
                 nStart = n0 * nTileSize;
          size_t mLen = std::min(mTileSize, mPad - mStart);
          size_t kLen = std::min(kTileSize, kPad - kStart);
          size_t nLen = std::min(nTileSize, nPad - nStart);

          Value aTile = rewriter.create<memref::SubViewOp>(
              loc, aMemArray,
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(mStart),
                                       rewriter.getIndexAttr(kStart)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(mLen),
                                       rewriter.getIndexAttr(kLen)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                       rewriter.getIndexAttr(1)});
          Value bTile = rewriter.create<memref::SubViewOp>(
              loc, bMemArray,
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(kStart),
                                       rewriter.getIndexAttr(nStart)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(kLen),
                                       rewriter.getIndexAttr(nLen)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                       rewriter.getIndexAttr(1)});
          Value cTile = rewriter.create<memref::SubViewOp>(
              loc, cMemArray,
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(mStart),
                                       rewriter.getIndexAttr(nStart)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(mLen),
                                       rewriter.getIndexAttr(nLen)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                       rewriter.getIndexAttr(1)});

          rewriter.create<buckyball::MatMulOp>(loc, aTile, bTile, cTile);
        }
      }
    }

    rewriter.eraseOp(tileMatMulOp);
    return success();
  }

private:
  int64_t lane, warp, bankDepth;
};

} // namespace

//===----------------------------------------------------------------------===//
// Tile Transpose Lowering Pattern
//===----------------------------------------------------------------------===//

namespace {

class TileTransposeLowering : public OpRewritePattern<tile::TileTransposeOp> {
  // Transpose needs both input and output in bank: 2 * rows * cols
  size_t computeBankRows(size_t rowsTileLen, size_t colsTileLen) const {
    return rowsTileLen * colsTileLen * 2;
  }

public:
  explicit TileTransposeLowering(MLIRContext *context, int64_t lane,
                                 int64_t /*warp*/, int64_t bankDepth,
                                 int64_t /*bankNum*/)
      : OpRewritePattern(context), lane(lane), bankDepth(bankDepth) {}

  LogicalResult matchAndRewrite(tile::TileTransposeOp tileTransposeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = tileTransposeOp.getLoc();

    Value inputMemArray = tileTransposeOp.getAMemArray();
    Value outputMemArray = tileTransposeOp.getBMemArray();

    auto inputType = cast<MemRefType>(inputMemArray.getType());
    auto outputType = cast<MemRefType>(outputMemArray.getType());
    auto inShape = inputType.getShape();
    auto outShape = outputType.getShape();

    size_t Rows = inShape[inShape.size() - 2];
    size_t Cols = inShape[inShape.size() - 1];

    if (outShape[outShape.size() - 2] != (int64_t)Cols ||
        outShape[outShape.size() - 1] != (int64_t)Rows)
      return tileTransposeOp.emitError(
          "Output shape must be transposed of input shape");

    const size_t rowMeta = lane, colMeta = lane;
    const size_t rowPad = ceilDiv(Rows, rowMeta) * rowMeta;
    const size_t colPad = ceilDiv(Cols, colMeta) * colMeta;

    size_t rowTileLen = 1, colTileLen = 1;

    while ((rowTileLen + 1) * rowMeta <= rowPad &&
           computeBankRows(rowTileLen + 1, colTileLen) <= (size_t)bankDepth)
      rowTileLen++;

    while ((colTileLen + 1) * colMeta <= colPad &&
           computeBankRows(rowTileLen, colTileLen + 1) <= (size_t)bankDepth)
      colTileLen++;

    const size_t rowTileSize = rowTileLen * rowMeta;
    const size_t colTileSize = colTileLen * colMeta;
    const size_t rowTileNum = ceilDiv(rowPad, rowTileSize);
    const size_t colTileNum = ceilDiv(colPad, colTileSize);

    for (size_t r0 = 0; r0 < rowTileNum; r0++) {
      for (size_t c0 = 0; c0 < colTileNum; c0++) {
        size_t rStart = r0 * rowTileSize, cStart = c0 * colTileSize;
        size_t rLen = std::min(rowTileSize, rowPad - rStart);
        size_t cLen = std::min(colTileSize, colPad - cStart);

        Value inTile = rewriter.create<memref::SubViewOp>(
            loc, inputMemArray,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(rStart),
                                     rewriter.getIndexAttr(cStart)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(rLen),
                                     rewriter.getIndexAttr(cLen)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                     rewriter.getIndexAttr(1)});
        Value outTile = rewriter.create<memref::SubViewOp>(
            loc, outputMemArray,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(cStart),
                                     rewriter.getIndexAttr(rStart)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(cLen),
                                     rewriter.getIndexAttr(rLen)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                     rewriter.getIndexAttr(1)});

        rewriter.create<buckyball::TransposeOp>(loc, inTile, outTile);
      }
    }

    rewriter.eraseOp(tileTransposeOp);
    return success();
  }

private:
  int64_t lane, bankDepth;
};

} // namespace

//===----------------------------------------------------------------------===//
// Tile Conv2d Lowering Pattern
//===----------------------------------------------------------------------===//

namespace {

class TileConv2dLowering : public OpRewritePattern<tile::TileConv2dOp> {
public:
  explicit TileConv2dLowering(MLIRContext *context, int64_t lane,
                              int64_t /*warp*/, int64_t bankDepth,
                              int64_t /*bankNum*/)
      : OpRewritePattern(context), lane(lane), bankDepth(bankDepth) {}

  LogicalResult matchAndRewrite(tile::TileConv2dOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = op.getInput();   // [N, H, W, C]
    Value filter = op.getFilter(); // [KH, KW, C, OC]
    Value output = op.getOutput(); // [N, OH, OW, OC]

    auto inType = cast<MemRefType>(input.getType());
    auto filterType = cast<MemRefType>(filter.getType());
    auto outType = cast<MemRefType>(output.getType());

    auto inShape = inType.getShape();
    auto fShape = filterType.getShape();
    auto outShape = outType.getShape();

    int64_t N = inShape[0], H = inShape[1], W = inShape[2], C = inShape[3];
    int64_t KH = fShape[0], KW = fShape[1], OC = fShape[3];
    int64_t OH = outShape[1], OW = outShape[2];

    IntegerType i64Type = rewriter.getI64Type();

    // Im2col patch dimensions: patchRows = OH*OW, patchCols = KH*KW*C
    int64_t patchCols = KH * KW * C;

    // Pad patchCols to lane boundary for matmul
    int64_t patchColsPad = ceilDiv(patchCols, (int64_t)lane) * lane;

    // Tile OH*OW dimension: how many output rows per tile
    // Each tile produces tileOHOW output pixels, requiring tileOHOW rows in patch matrix
    // Bank constraint: patch tile (tileOHOW * patchColsPad) must fit in bank
    int64_t tileOHOW = std::min((int64_t)bankDepth / std::max(patchColsPad, (int64_t)1),
                                (int64_t)(OH * OW));
    if (tileOHOW < 1) tileOHOW = 1;
    // Align to lane boundary
    tileOHOW = (tileOHOW / lane) * lane;
    if (tileOHOW < lane) tileOHOW = lane;

    int64_t totalOHOW = OH * OW;
    int64_t tileNum = ceilDiv(totalOHOW, tileOHOW);

    // For each batch
    for (int64_t n = 0; n < N; n++) {
      for (int64_t t = 0; t < tileNum; t++) {
        int64_t ohowStart = t * tileOHOW;
        int64_t ohowLen = std::min(tileOHOW, totalOHOW - ohowStart);

        // Compute start row/col in input space for im2col
        int64_t startRow = (ohowStart / OW);  // starting OH index
        int64_t startCol = (ohowStart % OW);  // starting OW index

        // Allocate temporary patch matrix: [ohowLen, patchColsPad]
        auto elemType = inType.getElementType();
        auto patchType = MemRefType::get({ohowLen, patchColsPad}, elemType);
        Value patchBuf = rewriter.create<memref::AllocOp>(loc, patchType);

        // Create subview of input for batch n, then collapse to 2D for im2col
        Value inBatch = rewriter.create<memref::SubViewOp>(
            loc, input,
            SmallVector<OpFoldResult>{
                rewriter.getIndexAttr(n), rewriter.getIndexAttr(0),
                rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)},
            SmallVector<OpFoldResult>{
                rewriter.getIndexAttr(1), rewriter.getIndexAttr(H),
                rewriter.getIndexAttr(W), rewriter.getIndexAttr(C)},
            SmallVector<OpFoldResult>{
                rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)});

        // Collapse [1, H, W, C] → [H, W*C] for im2col input
        auto collapseIn = rewriter.create<memref::CollapseShapeOp>(
            loc, inBatch,
            SmallVector<ReassociationIndices>{{0, 1}, {2, 3}});

        // Im2col: rearrange input patches into columns
        Value kRowVal = rewriter.create<arith::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(KH));
        Value kColVal = rewriter.create<arith::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(KW));
        Value inRowVal = rewriter.create<arith::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(H));
        Value inColVal = rewriter.create<arith::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(W * C));
        Value startRowVal = rewriter.create<arith::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(startRow));
        Value startColVal = rewriter.create<arith::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(startCol));

        rewriter.create<buckyball::Im2colOp>(
            loc, collapseIn, patchBuf,
            kRowVal, kColVal, inRowVal, inColVal, startRowVal, startColVal);

        // Reshape filter [KH, KW, C, OC] → [KH*KW*C, OC] for matmul
        Value filterReshaped = rewriter.create<memref::CollapseShapeOp>(
            loc, filter,
            SmallVector<ReassociationIndices>{{0, 1, 2}, {3}});

        // Create output subview for this tile: [ohowLen, OC]
        Value outBatch = rewriter.create<memref::SubViewOp>(
            loc, output,
            SmallVector<OpFoldResult>{
                rewriter.getIndexAttr(n), rewriter.getIndexAttr(0),
                rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)},
            SmallVector<OpFoldResult>{
                rewriter.getIndexAttr(1), rewriter.getIndexAttr(OH),
                rewriter.getIndexAttr(OW), rewriter.getIndexAttr(OC)},
            SmallVector<OpFoldResult>{
                rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)});

        // Collapse [1, OH, OW, OC] → [OH*OW, OC]
        auto collapseOut = rewriter.create<memref::CollapseShapeOp>(
            loc, outBatch,
            SmallVector<ReassociationIndices>{{0, 1, 2}, {3}});

        // Subview for the current tile rows
        Value outTile = rewriter.create<memref::SubViewOp>(
            loc, collapseOut,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(ohowStart),
                                     rewriter.getIndexAttr(0)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(ohowLen),
                                     rewriter.getIndexAttr(OC)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                     rewriter.getIndexAttr(1)});

        // MatMul: patch[ohowLen, patchCols] x filter[patchCols, OC] → out[ohowLen, OC]
        rewriter.create<buckyball::MatMulOp>(loc, patchBuf, filterReshaped,
                                            outTile);

        // Free temporary buffer
        rewriter.create<memref::DeallocOp>(loc, patchBuf);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t lane, bankDepth;
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Registration
//===----------------------------------------------------------------------===//

void populateLowerTileToBuckyballConversionPatterns(
    RewritePatternSet &patterns, int64_t lane, int64_t warp,
    int64_t bankDepth, int64_t bankNum) {
  patterns.add<TileMatMulLowering>(patterns.getContext(), lane, warp,
                                   bankDepth, bankNum);
  patterns.add<TileTransposeLowering>(patterns.getContext(), lane, warp,
                                     bankDepth, bankNum);
  patterns.add<TileConv2dLowering>(patterns.getContext(), lane, warp,
                                   bankDepth, bankNum);
}

//===----------------------------------------------------------------------===//
// LowerTileToBuckyball Pass
//===----------------------------------------------------------------------===//

namespace {
class LowerTileToBuckyballPass
    : public PassWrapper<LowerTileToBuckyballPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileToBuckyballPass)
  StringRef getArgument() const final { return "convert-tile-to-buckyball"; }
  StringRef getDescription() const final {
    return "Convert Tile dialect to Buckyball dialect";
  }
  LowerTileToBuckyballPass() = default;
  LowerTileToBuckyballPass(const LowerTileToBuckyballPass &) {}

  Option<int64_t> lane{*this, "lane",
                       llvm::cl::desc("Hardware lane width."),
                       llvm::cl::init(16)};
  Option<int64_t> warp{*this, "warp",
                       llvm::cl::desc("Warp depth."),
                       llvm::cl::init(16)};
  Option<int64_t> bankDepth{*this, "bank_depth",
                            llvm::cl::desc("Bank depth (rows per bank)."),
                            llvm::cl::init(4096)};
  Option<int64_t> bankNum{*this, "bank_num",
                          llvm::cl::desc("Number of banks."),
                          llvm::cl::init(8)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tile::TileDialect, buckyball::BuckyballDialect,
                    func::FuncDialect, memref::MemRefDialect,
                    arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LowerTileToBuckyballPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<buckyball::BuckyballDialect, memref::MemRefDialect,
                         arith::ArithDialect, scf::SCFDialect,
                         func::FuncDialect>();
  target.addIllegalDialect<tile::TileDialect>();

  RewritePatternSet patterns(context);
  populateLowerTileToBuckyballConversionPatterns(patterns, lane, warp,
                                                 bankDepth, bankNum);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerTileToBuckyballPass() {
  PassRegistration<LowerTileToBuckyballPass>();
}
} // namespace buddy
} // namespace mlir
