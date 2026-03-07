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
// Tile Matmul Lowering Pattern
//===----------------------------------------------------------------------===//

namespace {

class TileMatMulLowering : public OpRewritePattern<tile::TileMatMulOp> {
  // Compute tile's SPAD rows requirement
  size_t computeTileSpadRows(size_t mTileLen, size_t nTileLen, 
                             size_t kTileLen) const {
    size_t aMatrixRows = mTileLen * kTileLen;
    size_t bMatrixRows = kTileLen * nTileLen;
    return aMatrixRows + bMatrixRows;
  }

  // Compute tile's accumulator rows requirement
  size_t computeTileAccRows(size_t mTileLen, size_t nTileLen) const {
    return mTileLen * nTileLen;
  }

public:
  explicit TileMatMulLowering(MLIRContext *context, 
                              int64_t dim, int64_t spadRows, int64_t accRows,
                              int64_t warp, int64_t lane)
      : OpRewritePattern(context), dim(dim), spadRows(spadRows), 
        accRows(accRows), warp(warp), lane(lane) {}

  LogicalResult matchAndRewrite(tile::TileMatMulOp tileMatMulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = tileMatMulOp.getLoc();
    
    // Get input arrays
    Value aMemArray = tileMatMulOp.getAMemArray();
    Value bMemArray = tileMatMulOp.getBMemArray();
    Value cMemArray = tileMatMulOp.getCMemArray();
    
    auto aMemArrayType = cast<MemRefType>(aMemArray.getType());
    auto bMemArrayType = cast<MemRefType>(bMemArray.getType());
    
    IntegerType i64Type = rewriter.getI64Type();

    // Get A, B, C Matrix's shape - A[M][K], B[K][N], C[M][N]
    llvm::ArrayRef<int64_t> aMemArrayShape = aMemArrayType.getShape();
    llvm::ArrayRef<int64_t> bMemArrayShape = bMemArrayType.getShape();
    
    size_t M = aMemArrayShape[aMemArrayShape.size() - 2];
    size_t K = aMemArrayShape[aMemArrayShape.size() - 1];  
    size_t N = bMemArrayShape[bMemArrayShape.size() - 1];  

    // Tile's meta length (fixed by hardware)
    const size_t mMetaLen = lane;
    const size_t nMetaLen = lane;
    const size_t kMetaLen = warp;

    // Compute padded dimensions
    const size_t mPadded = ((M + mMetaLen - 1) / mMetaLen) * mMetaLen;
    const size_t nPadded = ((N + nMetaLen - 1) / nMetaLen) * nMetaLen;
    const size_t kPadded = ((K + kMetaLen - 1) / kMetaLen) * kMetaLen;

    // Compute tile lengths to maximize resource utilization
    const size_t maxSpadRows = spadRows / 2;  // Double buffer
    const size_t maxAccRows = accRows;
    
    size_t mTileLen = 1;
    size_t nTileLen = 1;
    size_t kTileLen = 1;
    
    // Extend N dimension
    bool increased = true;
    while (increased) {
      increased = false;
      size_t nextN = nTileLen + 1;
      size_t testSpadRows = computeTileSpadRows(mTileLen, nextN, kTileLen);
      size_t testAccRows = computeTileAccRows(mTileLen, nextN);
      if (testSpadRows <= maxSpadRows && testAccRows <= maxAccRows && 
          (nextN * nMetaLen) <= nPadded) {
        nTileLen = nextN;
        increased = true;
      }
    }
    
    // Extend M dimension
    increased = true;
    while (increased) {
      increased = false;
      size_t nextM = mTileLen + 1;
      size_t testSpadRows = computeTileSpadRows(nextM, nTileLen, kTileLen);
      size_t testAccRows = computeTileAccRows(nextM, nTileLen);
      if (testSpadRows <= maxSpadRows && testAccRows <= maxAccRows && 
          (nextM * mMetaLen) <= mPadded) {
        mTileLen = nextM;
        increased = true;
      }
    }

    // Compute tile parameters
    const size_t mTileSize = mTileLen * mMetaLen;
    const size_t nTileSize = nTileLen * nMetaLen;
    const size_t kTileSize = kTileLen * kMetaLen;
    
    const size_t mTileNum = (mPadded + mTileSize - 1) / mTileSize;
    const size_t nTileNum = (nPadded + nTileSize - 1) / nTileSize;
    const size_t kTileNum = (kPadded + kTileSize - 1) / kTileSize;
    
    // Last tile lengths (for handling non-perfectly-divisible dimensions)
    const size_t mLastTileLen = (mPadded % mTileSize == 0) ? mTileLen : 
                                (mPadded % mTileSize + mMetaLen - 1) / mMetaLen;
    const size_t nLastTileLen = (nPadded % nTileSize == 0) ? nTileLen : 
                                (nPadded % nTileSize + nMetaLen - 1) / nMetaLen;
    const size_t kLastTileLen = (kPadded % kTileSize == 0) ? kTileLen : 
                                (kPadded % kTileSize + kMetaLen - 1) / kMetaLen;
    
    // Generate tiled computation loops
    for (size_t k0 = 0; k0 < kTileNum; k0++) {
      for (size_t m0 = 0; m0 < mTileNum; m0++) {
        for (size_t n0 = 0; n0 < nTileNum; n0++) {
          // Determine current tile dimensions
          const bool isLastM = (m0 == mTileNum - 1);
          const bool isLastK = (k0 == kTileNum - 1);
          const bool isLastN = (n0 == nTileNum - 1);
          
          const size_t currentMLen = isLastM ? (mLastTileLen * mMetaLen) : mTileSize;
          const size_t currentKLen = isLastK ? (kLastTileLen * kMetaLen) : kTileSize;
          const size_t currentNLen = isLastN ? (nLastTileLen * nMetaLen) : nTileSize;

          // Calculate starting positions
          const size_t tileMStart = m0 * mTileSize;
          const size_t tileKStart = k0 * kTileSize;
          const size_t tileNStart = n0 * nTileSize;
          
          // Create subviews for current tile
          Value aTile = rewriter.create<memref::SubViewOp>(
              loc, aMemArray,
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(tileMStart), rewriter.getIndexAttr(tileKStart)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(currentMLen), rewriter.getIndexAttr(currentKLen)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)});
          Value bTile = rewriter.create<memref::SubViewOp>(
              loc, bMemArray,
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(tileKStart), rewriter.getIndexAttr(tileNStart)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(currentKLen), rewriter.getIndexAttr(currentNLen)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)});
          Value cTile = rewriter.create<memref::SubViewOp>(
              loc, cMemArray,
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(tileMStart), rewriter.getIndexAttr(tileNStart)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(currentMLen), rewriter.getIndexAttr(currentNLen)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)});

          // Create Buckyball MatMul operation for this tile
          rewriter.create<buckyball::MatMulOp>(loc, aTile, bTile, cTile);
        }
      }
    }

    rewriter.eraseOp(tileMatMulOp);
    return success();
  }

private:
  int64_t dim;
  int64_t spadRows;
  int64_t accRows;
  int64_t warp;
  int64_t lane;
};

} // namespace

namespace {

class TileTransposeLowering : public OpRewritePattern<tile::TileTransposeOp> {
  // Compute tile's SPAD rows requirement for transpose
  size_t computeTileSpadRows(size_t rowsTileLen, size_t colsTileLen) const {
    // Transpose requires both input and output tiles in SPAD
    // Input tile size: rowsTileLen * colsTileLen
    // Output tile size: colsTileLen * rowsTileLen (transposed)
    return rowsTileLen * colsTileLen * 2;  // Both input and output
  }

public:
  explicit TileTransposeLowering(MLIRContext *context, 
                                int64_t dim, int64_t spadRows, int64_t accRows,
                                int64_t warp, int64_t lane)
      : OpRewritePattern(context), dim(dim), spadRows(spadRows), 
        accRows(accRows), warp(warp), lane(lane) {}

  LogicalResult matchAndRewrite(tile::TileTransposeOp tileTransposeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = tileTransposeOp.getLoc();
    
    // Get input and output arrays
    Value inputMemArray = tileTransposeOp.getAMemArray();
    Value outputMemArray = tileTransposeOp.getBMemArray();
    
    auto inputMemArrayType = cast<MemRefType>(inputMemArray.getType());
    auto outputMemArrayType = cast<MemRefType>(outputMemArray.getType());
    
    IntegerType i64Type = rewriter.getI64Type();

    // Get input and output Matrix's shape - Input[Rows][Cols], Output[Cols][Rows]
    llvm::ArrayRef<int64_t> inputMemArrayShape = inputMemArrayType.getShape();
    llvm::ArrayRef<int64_t> outputMemArrayShape = outputMemArrayType.getShape();
    
    size_t Rows = inputMemArrayShape[inputMemArrayShape.size() - 2];
    size_t Cols = inputMemArrayShape[inputMemArrayShape.size() - 1];
    
    // Verify output shape is transposed
    if (outputMemArrayShape[outputMemArrayShape.size() - 2] != Cols ||
        outputMemArrayShape[outputMemArrayShape.size() - 1] != Rows) {
      return tileTransposeOp.emitError("Output shape must be transposed of input shape");
    }

    // Tile's meta length (fixed by hardware)
    const size_t rowMetaLen = lane;
    const size_t colMetaLen = lane;

    // Compute padded dimensions
    const size_t rowsPadded = ((Rows + rowMetaLen - 1) / rowMetaLen) * rowMetaLen;
    const size_t colsPadded = ((Cols + colMetaLen - 1) / colMetaLen) * colMetaLen;

    // Compute tile lengths to maximize resource utilization
    const size_t maxSpadRows = spadRows / 2;  // Double buffer
    
    size_t rowsTileLen = 1;
    size_t colsTileLen = 1;
    
    // Extend rows dimension
    bool increased = true;
    while (increased) {
      increased = false;
      size_t nextRows = rowsTileLen + 1;
      size_t testSpadRows = computeTileSpadRows(nextRows, colsTileLen);
      if (testSpadRows <= maxSpadRows && 
          (nextRows * rowMetaLen) <= rowsPadded) {
        rowsTileLen = nextRows;
        increased = true;
      }
    }
    
    // Extend cols dimension
    increased = true;
    while (increased) {
      increased = false;
      size_t nextCols = colsTileLen + 1;
      size_t testSpadRows = computeTileSpadRows(rowsTileLen, nextCols);
      if (testSpadRows <= maxSpadRows && 
          (nextCols * colMetaLen) <= colsPadded) {
        colsTileLen = nextCols;
        increased = true;
      }
    }

    // Compute tile parameters
    const size_t rowsTileSize = rowsTileLen * rowMetaLen;
    const size_t colsTileSize = colsTileLen * colMetaLen;
    
    const size_t rowsTileNum = (rowsPadded + rowsTileSize - 1) / rowsTileSize;
    const size_t colsTileNum = (colsPadded + colsTileSize - 1) / colsTileSize;
    
    // Last tile lengths (for handling non-perfectly-divisible dimensions)
    const size_t rowsLastTileLen = (rowsPadded % rowsTileSize == 0) ? rowsTileLen : 
                                  (rowsPadded % rowsTileSize + rowMetaLen - 1) / rowMetaLen;
    const size_t colsLastTileLen = (colsPadded % colsTileSize == 0) ? colsTileLen : 
                                  (colsPadded % colsTileSize + colMetaLen - 1) / colMetaLen;
    
    // Generate tiled computation loops
    for (size_t row0 = 0; row0 < rowsTileNum; row0++) {
      for (size_t col0 = 0; col0 < colsTileNum; col0++) {
        // Determine current tile dimensions
        const bool isLastRow = (row0 == rowsTileNum - 1);
        const bool isLastCol = (col0 == colsTileNum - 1);
        
        const size_t currentRowsLen = isLastRow ? (rowsLastTileLen * rowMetaLen) : rowsTileSize;
        const size_t currentColsLen = isLastCol ? (colsLastTileLen * colMetaLen) : colsTileSize;

        // Calculate starting positions
        const size_t tileRowStart = row0 * rowsTileSize;
        const size_t tileColStart = col0 * colsTileSize;
        
        // Create subviews for current input tile
        Value inputTile = rewriter.create<memref::SubViewOp>(
            loc, inputMemArray,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(tileRowStart), rewriter.getIndexAttr(tileColStart)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(currentRowsLen), rewriter.getIndexAttr(currentColsLen)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)});
        
        // Create subviews for current output tile (transposed)
        Value outputTile = rewriter.create<memref::SubViewOp>(
            loc, outputMemArray,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(tileColStart), rewriter.getIndexAttr(tileRowStart)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(currentColsLen), rewriter.getIndexAttr(currentRowsLen)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)});

        // Create Buckyball Transpose operation for this tile
        rewriter.create<buckyball::TransposeOp>(loc, inputTile, outputTile);
      }
    }

    rewriter.eraseOp(tileTransposeOp);
    return success();
  }

private:
  int64_t dim;
  int64_t spadRows;
  int64_t accRows;
  int64_t warp;
  int64_t lane;
};

} // namespace
void populateLowerTileToBuckyballConversionPatterns(
    RewritePatternSet &patterns, int64_t dim, int64_t spadRows, 
    int64_t accRows, int64_t warp, int64_t lane) {
  patterns.add<TileMatMulLowering>(patterns.getContext(), dim, spadRows, 
                                   accRows, warp, lane);
  patterns.add<TileTransposeLowering>(patterns.getContext(), dim, spadRows,
                                     accRows, warp, lane);
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

  Option<int64_t> dim{*this, "dim", 
                      llvm::cl::desc("Size of Scratchpad line."),
                      llvm::cl::init(16)};
  Option<int64_t> spadRows{*this, "spad_rows",
                           llvm::cl::desc("The row of spad."),
                           llvm::cl::init(1024)};
  Option<int64_t> accRows{*this, "acc_rows", 
                          llvm::cl::desc("The row of acc."),
                          llvm::cl::init(1024)};
  Option<int64_t> warp{*this, "warp", 
                       llvm::cl::desc("Size of warp."),
                       llvm::cl::init(16)};
  Option<int64_t> lane{*this, "lane", 
                       llvm::cl::desc("Size of lane."),
                       llvm::cl::init(16)};

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
                         arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
  target.addIllegalDialect<tile::TileDialect>();
  
  RewritePatternSet patterns(context);
  populateLowerTileToBuckyballConversionPatterns(patterns, dim, spadRows, 
                                                 accRows, warp, lane);
  
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

