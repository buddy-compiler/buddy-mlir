//=== LowerLinalgToXTAME.cpp - Linalg to XTAME Dialect Lowering Pass --===//
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
// This file defines Linalg dialect lowering pass to XTAME dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

#include "Dialect/XTAME/XTAMEDialect.h"
#include "Dialect/XTAME/XTAMEOps.h"

using namespace mlir;
using namespace buddy::xtame;

namespace {

// One matrix register row is RLEN = 128 bit, so a reduction chunk holds at most
// 16 i8 elements; 4 i8 elements is the smallest chunk that keeps a whole i32
// accumulator element in the store path.
constexpr int64_t kMaxChunkBytes = 16;
constexpr int64_t kSmallChunkBytes = 4;

// The K extent is reduced with 16-element chunks while at least 16 elements
// remain, then with 4-element chunks, then with a peeled remainder below four.
// Splitting this way keeps the number of emitted loops at three whatever K is,
// and still uses the wide chunk when K is not a multiple of 16 (K = 20 becomes
// 16 + 4 instead of five 4-element chunks).
struct KPartition {
  int64_t wide;   // elements covered by 16-element chunks, a multiple of 16
  int64_t narrow; // elements covered by 4-element chunks: 0, 4, 8 or 12
  int64_t tail;   // peeled elements: 0 to 3
  int64_t steps() const {
    return wide / kMaxChunkBytes + narrow / kSmallChunkBytes +
           (tail > 0 ? 1 : 0);
  }
};

static KPartition partitionK(int64_t k) {
  KPartition part;
  part.tail = k % kSmallChunkBytes;
  const int64_t full = k - part.tail;
  part.wide = (full / kMaxChunkBytes) * kMaxChunkBytes;
  part.narrow = full - part.wide;
  return part;
}

// Emit the AME sequence for one static MxK i8 times KxN i8 matmul into an i32
// accumulator. `m` must be at least four rows. The operands must be contiguous
// or contiguous slices of a contiguous buffer: the row strides follow from the
// extents, and only the base pointer of each slice varies.
static void lowerMatmul(PatternRewriter &rewriter, Location loc, Value a,
                        Value b, Value c, int64_t m, int64_t k, int64_t n) {
  auto createImmediate = [&](StringRef name, StringRef attr, int64_t value) {
    OperationState state(loc, name);
    state.addAttribute(attr, rewriter.getI64IntegerAttr(value));
    rewriter.create(state);
  };

  auto createLoad = [&](StringRef name, int64_t reg, int64_t stride,
                        Value base) {
    Value strideValue =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                  rewriter.getI64IntegerAttr(stride));

    OperationState state(loc, name);
    state.addAttribute("md", rewriter.getI64IntegerAttr(reg));
    state.addOperands(strideValue);
    state.addOperands(base);
    rewriter.create(state);
  };

  // tilek is the byte width of the innermost dimension of the operation, not
  // an element count: one i32 C element is 4 bytes, and an i8 reduction tile
  // of K elements is K bytes.
  constexpr int64_t maxTileKBytes = 16; // RLEN = 128 bit
  const int64_t kBytes = k;             // A and B are i8 here.
  const int64_t aStride = k;            // bytes per row of A
  const int64_t bStride = n;            // bytes per row of B
  const int64_t cStride = n * 4;        // bytes per row of C

  // The accumulator tile is four columns wide; wider outputs use blocks.
  createImmediate("xt_ame.th.mcfgni", "tilen", 4);

  const bool singleK = kBytes <= maxTileKBytes;
  // A does not depend on the output columns and the multiply-accumulate
  // leaves ms1 untouched, so a single K extent can be loaded once.
  const bool hoistA = singleK && n > 4;

  // Partition the output rows into tiles of four to sixteen rows.
  constexpr int64_t maxTileRows = 16;
  SmallVector<int64_t> rowTiles;
  for (int64_t remaining = m; remaining > 0;) {
    int64_t rows = remaining < maxTileRows ? remaining : maxTileRows;
    if (remaining - rows > 0 && remaining - rows < 4)
      rows = remaining - 4;
    rowTiles.push_back(rows);
    remaining -= rows;
  }

  int64_t rowOffset = 0;
  for (int64_t rows : rowTiles) {
    Value aTile = a;
    if (rowOffset != 0 || rows != m) {
      auto oneRow = rewriter.getIndexAttr(1);
      SmallVector<OpFoldResult> offsetsA{rewriter.getIndexAttr(rowOffset),
                                         rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> sizesA{rewriter.getIndexAttr(rows),
                                       rewriter.getIndexAttr(k)};
      SmallVector<OpFoldResult> stridesA{oneRow, oneRow};
      aTile = memref::SubViewOp::create(rewriter, loc, a, offsetsA, sizesA,
                                        stridesA);
    }
    createImmediate("xt_ame.th.mcfgmi", "tilem", rows);

    auto createMma = [&](int64_t md) {
      OperationState mmaState(loc, "xt_ame.th.mmacc.w.b");
      mmaState.addAttribute("md", rewriter.getI64IntegerAttr(md));
      mmaState.addAttribute("ms2", rewriter.getI64IntegerAttr(2));
      mmaState.addAttribute("ms1", rewriter.getI64IntegerAttr(1));
      rewriter.create(mmaState);
    };

    // Reduce over the K extent into the accumulator held in m0.
    auto emitReduction = [&](Value bTile, int64_t cols) {
      if (singleK) {
        // A single multiply-accumulate covers the whole K extent.
        createImmediate("xt_ame.th.mcfgki", "tilek", kBytes);
        if (!hoistA)
          createLoad("xt_ame.th.mlde8", 1, aStride, aTile);
        createLoad("xt_ame.th.mldte8", 2, bStride, bTile);
        createMma(0);
        return;
      }

      // K exceeds one register row: 16-element chunks while they fit, then
      // 4-element chunks, then the remainder below four.
      auto one = rewriter.getIndexAttr(1);
      SmallVector<OpFoldResult> strides{one, one};

      auto emitChunkBody = [&](OpFoldResult kOffset, int64_t bytes) {
        SmallVector<OpFoldResult> offsetsA{rewriter.getIndexAttr(0), kOffset};
        SmallVector<OpFoldResult> offsetsB{kOffset, rewriter.getIndexAttr(0)};
        SmallVector<OpFoldResult> sizesA{rewriter.getIndexAttr(rows),
                                         rewriter.getIndexAttr(bytes)};
        SmallVector<OpFoldResult> sizesB{rewriter.getIndexAttr(bytes),
                                         rewriter.getIndexAttr(cols)};

        Value tileA = memref::SubViewOp::create(rewriter, loc, aTile, offsetsA,
                                                sizesA, strides);
        Value tileB = memref::SubViewOp::create(rewriter, loc, bTile, offsetsB,
                                                sizesB, strides);

        createLoad("xt_ame.th.mlde8", 1, aStride, tileA);
        createLoad("xt_ame.th.mldte8", 2, bStride, tileB);
        createMma(0);
      };

      // The tile width is constant inside the loop, so configure it once.
      auto emitChunkLoop = [&](int64_t lower, int64_t upper, int64_t bytes) {
        createImmediate("xt_ame.th.mcfgki", "tilek", bytes);
        Value lowerV = arith::ConstantIndexOp::create(rewriter, loc, lower);
        Value upperV = arith::ConstantIndexOp::create(rewriter, loc, upper);
        Value stepV = arith::ConstantIndexOp::create(rewriter, loc, bytes);
        auto kLoop = scf::ForOp::create(rewriter, loc, lowerV, upperV, stepV);
        rewriter.setInsertionPointToStart(kLoop.getBody());
        emitChunkBody(kLoop.getInductionVar(), bytes);
        rewriter.setInsertionPointAfter(kLoop);
      };

      const KPartition part = partitionK(k);
      if (part.wide > 0)
        emitChunkLoop(0, part.wide, kMaxChunkBytes);
      if (part.narrow > 0)
        emitChunkLoop(part.wide, part.wide + part.narrow, kSmallChunkBytes);
      if (part.tail > 0) {
        createImmediate("xt_ame.th.mcfgki", "tilek", part.tail);
        emitChunkBody(rewriter.getIndexAttr(part.wide + part.narrow),
                      part.tail);
      }
    };

    // Emit load, reduce, and store for one block of `cols` output columns.
    auto emitColumnBlock = [&](Value cTile, Value bTile, int64_t cols) {
      const int64_t rowBytes = 4 * cols;

      // Preserve the initial output tile: linalg.matmul accumulates into C.
      createImmediate("xt_ame.th.mcfgki", "tilek", rowBytes);
      createLoad("xt_ame.th.mlde32", 0, cStride, cTile);

      emitReduction(bTile, cols);

      // Restore the block row width before storing.
      createImmediate("xt_ame.th.mcfgki", "tilek", rowBytes);

      Value strideC =
          arith::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                    rewriter.getI64IntegerAttr(cStride));

      OperationState storeState(loc, "xt_ame.th.mste32");
      storeState.addAttribute("ms3", rewriter.getI64IntegerAttr(0));
      storeState.addOperands(strideC);
      storeState.addOperands(cTile);
      rewriter.create(storeState);
    };

    if (hoistA) {
      createImmediate("xt_ame.th.mcfgki", "tilek", kBytes);
      createLoad("xt_ame.th.mlde8", 1, aStride, aTile);
    }

    const int64_t colTail = n % 4;
    const int64_t colFull = n - colTail;
    auto oneIndex = rewriter.getIndexAttr(1);
    SmallVector<OpFoldResult> blockStrides{oneIndex, oneIndex};

    // Emit one column block; `columnOffset` is either a constant or the
    // induction variable of the column loop. A block that covers the whole
    // output uses the operands directly.
    auto emitColumnRange = [&](int64_t cols, OpFoldResult columnOffset,
                               bool wholeOperand) {
      if (wholeOperand) {
        emitColumnBlock(c, b, cols);
        return;
      }

      SmallVector<OpFoldResult> offsetsC{rewriter.getIndexAttr(rowOffset),
                                         columnOffset};
      SmallVector<OpFoldResult> offsetsB{rewriter.getIndexAttr(0),
                                         columnOffset};
      SmallVector<OpFoldResult> sizesC{rewriter.getIndexAttr(rows),
                                       rewriter.getIndexAttr(cols)};
      SmallVector<OpFoldResult> sizesB{rewriter.getIndexAttr(k),
                                       rewriter.getIndexAttr(cols)};

      Value cTile = memref::SubViewOp::create(rewriter, loc, c, offsetsC,
                                              sizesC, blockStrides);
      Value bTile = memref::SubViewOp::create(rewriter, loc, b, offsetsB,
                                              sizesB, blockStrides);
      emitColumnBlock(cTile, bTile, cols);
    };

    struct ColumnBlock {
      int64_t offset;
      int64_t cols;
    };
    SmallVector<ColumnBlock> columnBlocks;
    for (int64_t column = 0; column < colFull; column += 4)
      columnBlocks.push_back({column, 4});
    if (colTail > 0)
      columnBlocks.push_back({colFull, colTail});

    const KPartition kPartition = partitionK(k);
    const int64_t kSteps = singleK ? 1 : kPartition.steps();
    const bool interchange = kSteps > 1 && columnBlocks.size() <= 6;
    if (interchange) {
      // Loop interchange: load A once per K chunk and accumulate every column
      // block, which keeps its accumulator in a register of its own. m1 and
      // m2 stay reserved for A and B.
      static const int64_t accRegs[] = {0, 3, 4, 5, 6, 7};
      SmallVector<Value> cTiles;
      SmallVector<Value> bTiles;

      for (size_t index = 0; index < columnBlocks.size(); ++index) {
        const ColumnBlock &block = columnBlocks[index];
        OpFoldResult columnOffset = rewriter.getIndexAttr(block.offset);
        bool wholeOperand =
            rowOffset == 0 && rows == m && block.offset == 0 && block.cols == n;
        if (wholeOperand) {
          cTiles.push_back(c);
          bTiles.push_back(b);
        } else {
          SmallVector<OpFoldResult> offsetsC{rewriter.getIndexAttr(rowOffset),
                                             columnOffset};
          SmallVector<OpFoldResult> offsetsB{rewriter.getIndexAttr(0),
                                             columnOffset};
          SmallVector<OpFoldResult> sizesC{rewriter.getIndexAttr(rows),
                                           rewriter.getIndexAttr(block.cols)};
          SmallVector<OpFoldResult> sizesB{rewriter.getIndexAttr(k),
                                           rewriter.getIndexAttr(block.cols)};
          cTiles.push_back(memref::SubViewOp::create(rewriter, loc, c, offsetsC,
                                                     sizesC, blockStrides));
          bTiles.push_back(memref::SubViewOp::create(rewriter, loc, b, offsetsB,
                                                     sizesB, blockStrides));
        }

        createImmediate("xt_ame.th.mcfgki", "tilek", 4 * block.cols);
        createLoad("xt_ame.th.mlde32", accRegs[index], cStride, cTiles[index]);
      }

      // Full K chunks run in a loop so the code does not grow with K; the
      // remainder is peeled. The accumulators stay live across iterations.
      auto emitChunkBody = [&](OpFoldResult kOffset, int64_t bytes) {
        SmallVector<OpFoldResult> offsetsA{rewriter.getIndexAttr(0), kOffset};
        SmallVector<OpFoldResult> sizesA{rewriter.getIndexAttr(rows),
                                         rewriter.getIndexAttr(bytes)};
        Value aSlice = memref::SubViewOp::create(rewriter, loc, aTile, offsetsA,
                                                 sizesA, blockStrides);

        createLoad("xt_ame.th.mlde8", 1, aStride, aSlice);

        for (size_t index = 0; index < columnBlocks.size(); ++index) {
          SmallVector<OpFoldResult> offsetsB{kOffset, rewriter.getIndexAttr(0)};
          SmallVector<OpFoldResult> sizesB{
              rewriter.getIndexAttr(bytes),
              rewriter.getIndexAttr(columnBlocks[index].cols)};
          Value bSlice = memref::SubViewOp::create(
              rewriter, loc, bTiles[index], offsetsB, sizesB, blockStrides);
          createLoad("xt_ame.th.mldte8", 2, bStride, bSlice);
          createMma(accRegs[index]);
        }
      };

      // The tile width is constant inside the loop, so configure it once.
      auto emitChunkLoop = [&](int64_t lower, int64_t upper, int64_t bytes) {
        createImmediate("xt_ame.th.mcfgki", "tilek", bytes);
        Value lowerV = arith::ConstantIndexOp::create(rewriter, loc, lower);
        Value upperV = arith::ConstantIndexOp::create(rewriter, loc, upper);
        Value stepV = arith::ConstantIndexOp::create(rewriter, loc, bytes);
        auto kLoop = scf::ForOp::create(rewriter, loc, lowerV, upperV, stepV);
        rewriter.setInsertionPointToStart(kLoop.getBody());
        emitChunkBody(kLoop.getInductionVar(), bytes);
        rewriter.setInsertionPointAfter(kLoop);
      };

      if (kPartition.wide > 0)
        emitChunkLoop(0, kPartition.wide, kMaxChunkBytes);
      if (kPartition.narrow > 0)
        emitChunkLoop(kPartition.wide, kPartition.wide + kPartition.narrow,
                      kSmallChunkBytes);
      if (kPartition.tail > 0) {
        createImmediate("xt_ame.th.mcfgki", "tilek", kPartition.tail);
        emitChunkBody(
            rewriter.getIndexAttr(kPartition.wide + kPartition.narrow),
            kPartition.tail);
      }

      for (size_t index = 0; index < columnBlocks.size(); ++index) {
        createImmediate("xt_ame.th.mcfgki", "tilek",
                        4 * columnBlocks[index].cols);
        Value strideC =
            arith::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                      rewriter.getI64IntegerAttr(cStride));
        OperationState storeState(loc, "xt_ame.th.mste32");
        storeState.addAttribute("ms3",
                                rewriter.getI64IntegerAttr(accRegs[index]));
        storeState.addOperands(strideC);
        storeState.addOperands(cTiles[index]);
        rewriter.create(storeState);
      }
    } else if (colFull >= 8) {
      Value lower = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value upper = arith::ConstantIndexOp::create(rewriter, loc, colFull);
      Value step = arith::ConstantIndexOp::create(rewriter, loc, 4);

      auto jLoop = scf::ForOp::create(rewriter, loc, lower, upper, step);
      rewriter.setInsertionPointToStart(jLoop.getBody());
      emitColumnRange(4, jLoop.getInductionVar(), /*wholeOperand=*/false);
      rewriter.setInsertionPointAfter(jLoop);
    } else if (colFull == 4) {
      emitColumnRange(4, rewriter.getIndexAttr(0),
                      /*wholeOperand=*/rowOffset == 0 && rows == m && n == 4);
    }

    if (!interchange && colTail > 0)
      emitColumnRange(colTail, rewriter.getIndexAttr(colFull),
                      /*wholeOperand=*/rowOffset == 0 && rows == m &&
                          colFull == 0);

    rowOffset += rows;
  }
}

// Fill M, K and N when the operands are the supported static i8 and i32
// memrefs of a rank-2 matmul. Identity layouts are required because the
// lowering derives the row strides from the extents.
static bool matmulShape(Value a, Value b, Value c, int64_t &m, int64_t &k,
                        int64_t &n) {
  auto aType = dyn_cast<MemRefType>(a.getType());
  auto bType = dyn_cast<MemRefType>(b.getType());
  auto cType = dyn_cast<MemRefType>(c.getType());
  if (!aType || !bType || !cType || !aType.hasStaticShape() ||
      !bType.hasStaticShape() || !cType.hasStaticShape() ||
      aType.getRank() != 2 || bType.getRank() != 2 || cType.getRank() != 2)
    return false;
  m = aType.getShape()[0];
  k = aType.getShape()[1];
  n = bType.getShape()[1];
  // A single tile register holds up to 16 rows and the i32 accumulator row is
  // one register wide, so the output is processed in blocks of up to four
  // columns and up to 16 rows. The model also requires at least four rows.
  if (m < 4 || k < 1 || n < 1)
    return false;
  MLIRContext *context = a.getContext();
  return a.getType() == MemRefType::get({m, k}, IntegerType::get(context, 8)) &&
         b.getType() == MemRefType::get({k, n}, IntegerType::get(context, 8)) &&
         c.getType() == MemRefType::get({m, n}, IntegerType::get(context, 32));
}

// Stack the zero-point correction needs for one quantized matmul: the row-sum,
// column-sum and column-term buffers. The matrices of ones are module
// constants.
static int64_t correctionBytes(int64_t m, int64_t k, int64_t n) {
  (void)k;
  return 4 * m + 20 * n;
}

// Fill a static rank-2 memref with a constant scalar.
static void emitFill(PatternRewriter &rewriter, Location loc, Value memref,
                     Value value, int64_t rows, int64_t cols) {
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value rowBound = arith::ConstantIndexOp::create(rewriter, loc, rows);
  Value colBound = arith::ConstantIndexOp::create(rewriter, loc, cols);
  auto outer = scf::ForOp::create(rewriter, loc, zero, rowBound, one);
  rewriter.setInsertionPointToStart(outer.getBody());
  auto inner = scf::ForOp::create(rewriter, loc, zero, colBound, one);
  rewriter.setInsertionPointToStart(inner.getBody());
  memref::StoreOp::create(
      rewriter, loc, value, memref,
      ValueRange{outer.getInductionVar(), inner.getInductionVar()});
  rewriter.setInsertionPointAfter(inner);
  rewriter.setInsertionPointAfter(outer);
}

// Materialise a static matrix of ones as a private constant of the module, so
// the zero-point correction does not fill it at run time. One symbol per shape
// and module; the AME load reads it through the usual descriptor path.
static Value getOnesConstant(PatternRewriter &rewriter, Location loc,
                             Operation *anchor, ArrayRef<int64_t> shape) {
  auto memrefType = MemRefType::get(shape, rewriter.getI8Type());
  std::string name = "xtame_ones";
  for (int64_t dim : shape)
    name += "_" + std::to_string(dim);

  ModuleOp module = anchor->getParentOfType<ModuleOp>();
  if (!module.lookupSymbol<memref::GlobalOp>(name)) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto tensorType = RankedTensorType::get(shape, rewriter.getI8Type());
    auto init =
        DenseElementsAttr::get(tensorType, rewriter.getI8IntegerAttr(1));
    memref::GlobalOp::create(rewriter, loc, name,
                             rewriter.getStringAttr("private"), memrefType,
                             init,
                             /*constant=*/true, rewriter.getI64IntegerAttr(16));
  }
  return memref::GetGlobalOp::create(rewriter, loc, memrefType, name);
}

// linalg.quantized_matmul accumulates (A - AZp) * (B - BZp). The matmul above
// leaves sum_k A*B, so every output element still needs
//   -AZp * sum_k B - BZp * sum_k A + K * AZp * BZp.
// Both sums are themselves matrix products with a vector of ones, so they use
// the same AME sequence as the matmul: A times a Kx1 column of ones gives the
// row sums, and a 4xK block of ones times B gives the column sums in its first
// row (a tile needs at least four rows). Only the final update of C is scalar.
static void emitZeroPointCorrection(PatternRewriter &rewriter, Location loc,
                                    Operation *anchor, Value a, Value b,
                                    Value c, Value aZp, Value bZp, int64_t m,
                                    int64_t k, int64_t n) {
  auto i32 = rewriter.getI32Type();
  Value aOnes = getOnesConstant(rewriter, loc, anchor, {k, 1});
  Value bOnes = getOnesConstant(rewriter, loc, anchor, {4, k});
  Value rowSums =
      memref::AllocaOp::create(rewriter, loc, MemRefType::get({m, 1}, i32));
  Value colSums =
      memref::AllocaOp::create(rewriter, loc, MemRefType::get({4, n}, i32));
  Value colTerm =
      memref::AllocaOp::create(rewriter, loc, MemRefType::get({n}, i32));

  Value zero32 = arith::ConstantOp::create(rewriter, loc, i32,
                                           rewriter.getI32IntegerAttr(0));
  emitFill(rewriter, loc, rowSums, zero32, m, 1);
  emitFill(rewriter, loc, colSums, zero32, 4, n);

  // rowSums[i, 0] = sum_k A[i, k] and colSums[0, l] = sum_k B[k, l].
  lowerMatmul(rewriter, loc, a, aOnes, rowSums, m, k, 1);
  lowerMatmul(rewriter, loc, bOnes, b, colSums, 4, k, n);

  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value mBound = arith::ConstantIndexOp::create(rewriter, loc, m);
  Value nBound = arith::ConstantIndexOp::create(rewriter, loc, n);

  // AZp * colSums[0, l] once per output column instead of once per element.
  auto cols = scf::ForOp::create(rewriter, loc, zero, nBound, one);
  rewriter.setInsertionPointToStart(cols.getBody());
  {
    Value col = cols.getInductionVar();
    Value sum =
        memref::LoadOp::create(rewriter, loc, colSums, ValueRange{zero, col});
    Value term = arith::MulIOp::create(rewriter, loc, aZp, sum);
    memref::StoreOp::create(rewriter, loc, term, colTerm, ValueRange{col});
  }
  rewriter.setInsertionPointAfter(cols);

  // C[i, l] += K * AZp * BZp - BZp * rowSums[i, 0] - AZp * colSums[0, l]
  Value kValue = arith::ConstantIntOp::create(rewriter, loc, k, 32);
  Value constant = arith::MulIOp::create(rewriter, loc, kValue, aZp);
  constant = arith::MulIOp::create(rewriter, loc, constant, bZp);
  auto outer = scf::ForOp::create(rewriter, loc, zero, mBound, one);
  rewriter.setInsertionPointToStart(outer.getBody());
  Value row = outer.getInductionVar();
  Value rowSum =
      memref::LoadOp::create(rewriter, loc, rowSums, ValueRange{row, zero});
  Value rowTerm = arith::MulIOp::create(rewriter, loc, bZp, rowSum);
  auto inner = scf::ForOp::create(rewriter, loc, zero, nBound, one);
  rewriter.setInsertionPointToStart(inner.getBody());
  {
    Value col = inner.getInductionVar();
    Value colPart =
        memref::LoadOp::create(rewriter, loc, colTerm, ValueRange{col});
    Value current =
        memref::LoadOp::create(rewriter, loc, c, ValueRange{row, col});
    Value value = arith::SubIOp::create(rewriter, loc, current, rowTerm);
    value = arith::SubIOp::create(rewriter, loc, value, colPart);
    value = arith::AddIOp::create(rewriter, loc, value, constant);
    memref::StoreOp::create(rewriter, loc, value, c, ValueRange{row, col});
  }
  rewriter.setInsertionPointAfter(inner);
  rewriter.setInsertionPointAfter(outer);
}

class MatMulToXTAMELowering : public OpConversionPattern<linalg::MatmulOp> {
public:
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, linalg::MatmulOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor; // Unused, but required for the interface
    if (!op.hasPureBufferSemantics())
      return rewriter.notifyMatchFailure(op, "expected buffer semantics");
    Value a = op.getDpsInputOperand(0)->get();
    Value b = op.getDpsInputOperand(1)->get();
    Value c = op.getDpsInitOperand(0)->get();
    int64_t m, k, n;
    if (!matmulShape(a, b, c, m, k, n))
      return rewriter.notifyMatchFailure(
          op, "unsupported linalg.matmul: expected static identity-layout "
              "memrefs of i8 times i8 with an i32 accumulator and at least "
              "four rows");
    lowerMatmul(rewriter, op.getLoc(), a, b, c, m, k, n);
    rewriter.eraseOp(op);
    return success();
  }
};

class BatchMatmulToXTAMELowering
    : public OpConversionPattern<linalg::BatchMatmulOp> {
public:
  using OpConversionPattern<linalg::BatchMatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::BatchMatmulOp op,
                  linalg::BatchMatmulOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor; // Unused, but required for the interface
    if (!op.hasPureBufferSemantics())
      return rewriter.notifyMatchFailure(op, "expected buffer semantics");
    Value a = op.getDpsInputOperand(0)->get();
    Value b = op.getDpsInputOperand(1)->get();
    Value c = op.getDpsInitOperand(0)->get();
    auto aType = dyn_cast<MemRefType>(a.getType());
    auto bType = dyn_cast<MemRefType>(b.getType());
    auto cType = dyn_cast<MemRefType>(c.getType());
    if (!aType || !bType || !cType || !aType.hasStaticShape() ||
        !bType.hasStaticShape() || !cType.hasStaticShape() ||
        aType.getRank() != 3 || bType.getRank() != 3 || cType.getRank() != 3)
      return rewriter.notifyMatchFailure(
          op,
          "unsupported linalg.batch_matmul: expected static rank-3 memrefs");
    const int64_t batch = aType.getShape()[0];
    const int64_t m = aType.getShape()[1];
    const int64_t k = aType.getShape()[2];
    const int64_t n = bType.getShape()[2];
    if (batch < 1 || m < 4 || k < 1 || n < 1)
      return rewriter.notifyMatchFailure(
          op, "unsupported linalg.batch_matmul: every matmul needs at least "
              "four rows");
    MLIRContext *context = a.getContext();
    if (a.getType() !=
            MemRefType::get({batch, m, k}, IntegerType::get(context, 8)) ||
        b.getType() !=
            MemRefType::get({batch, k, n}, IntegerType::get(context, 8)) ||
        c.getType() !=
            MemRefType::get({batch, m, n}, IntegerType::get(context, 32)))
      return rewriter.notifyMatchFailure(
          op, "unsupported linalg.batch_matmul: expected i8 inputs, an i32 "
              "accumulator and identity layouts");

    Location loc = op.getLoc();
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value bound = arith::ConstantIndexOp::create(rewriter, loc, batch);
    auto loop = scf::ForOp::create(rewriter, loc, zero, bound, one);
    rewriter.setInsertionPointToStart(loop.getBody());
    Value index = loop.getInductionVar();

    // A batch slice is contiguous, so dropping the unit batch dimension leaves
    // the row strides the matmul lowering expects. The result type has to be
    // rank-reduced explicitly: the plain subview builder keeps the source rank.
    auto slice = [&](Value source, int64_t rows, int64_t cols) {
      SmallVector<OpFoldResult> offsets{index, rewriter.getIndexAttr(0),
                                        rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(1),
                                      rewriter.getIndexAttr(rows),
                                      rewriter.getIndexAttr(cols)};
      SmallVector<OpFoldResult> strides(3, rewriter.getIndexAttr(1));
      MemRefType type = memref::SubViewOp::inferRankReducedResultType(
          {rows, cols}, cast<MemRefType>(source.getType()), offsets, sizes,
          strides);
      return memref::SubViewOp::create(rewriter, loc, type, source, offsets,
                                       sizes, strides);
    };
    Value aSlice = slice(a, m, k);
    Value bSlice = slice(b, k, n);
    Value cSlice = slice(c, m, n);
    lowerMatmul(rewriter, loc, aSlice, bSlice, cSlice, m, k, n);
    rewriter.setInsertionPointAfter(loop);
    rewriter.eraseOp(op);
    return success();
  }
};

class QuantizedMatmulToXTAMELowering
    : public OpConversionPattern<linalg::QuantizedMatmulOp> {
public:
  QuantizedMatmulToXTAMELowering(MLIRContext *context, int64_t budget)
      : OpConversionPattern(context), temporaryBudget(budget) {};

  LogicalResult
  matchAndRewrite(linalg::QuantizedMatmulOp op,
                  linalg::QuantizedMatmulOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor; // Unused, but required for the interface
    if (!op.hasPureBufferSemantics())
      return rewriter.notifyMatchFailure(op, "expected buffer semantics");
    Value a = op.getDpsInputOperand(0)->get();
    Value b = op.getDpsInputOperand(1)->get();
    Value aZp = op.getDpsInputOperand(2)->get();
    Value bZp = op.getDpsInputOperand(3)->get();
    Value c = op.getDpsInitOperand(0)->get();
    int64_t m, k, n;
    if (!matmulShape(a, b, c, m, k, n))
      return rewriter.notifyMatchFailure(
          op, "unsupported linalg.quantized_matmul: expected static "
              "identity-layout memrefs of i8 times i8 with an i32 accumulator "
              "and at least four rows");
    if (!aZp.getType().isInteger(32) || !bZp.getType().isInteger(32))
      return rewriter.notifyMatchFailure(
          op, "unsupported linalg.quantized_matmul: expected i32 zero points");

    // The correction keeps five temporaries on the stack: the two matrices of
    // ones and the three i32 buffers. Reject a shape whose temporaries would
    // not fit in the budget instead of emitting a program that overflows the
    // stack at run time.
    const int64_t temporaryBytes = correctionBytes(m, k, n);
    if (temporaryBytes > temporaryBudget)
      return rewriter.notifyMatchFailure(
          op, "the zero-point correction needs " +
                  std::to_string(temporaryBytes) +
                  " bytes of stack temporaries, above the " +
                  std::to_string(temporaryBudget) +
                  " byte limit (raise --max-temporary-bytes)");

    Location loc = op.getLoc();
    lowerMatmul(rewriter, loc, a, b, c, m, k, n);
    emitZeroPointCorrection(rewriter, loc, op, a, b, c, aZp, bZp, m, k, n);
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t temporaryBudget;
};

class LowerLinalgToXTAMEPass
    : public PassWrapper<LowerLinalgToXTAMEPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToXTAMEPass)

  LowerLinalgToXTAMEPass() = default;
  LowerLinalgToXTAMEPass(const LowerLinalgToXTAMEPass &) {}

  StringRef getArgument() const final { return "lower-linalg-to-xtame"; }

  StringRef getDescription() const final {
    return "Lower linalg matmul to XTAME operations.";
  }

  Option<int64_t> maxTemporaryBytes{
      *this, "max-temporary-bytes",
      llvm::cl::desc("Reject a linalg.quantized_matmul whose zero-point "
                     "correction would need more than this many bytes of stack "
                     "temporaries"),
      llvm::cl::init(16384)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<buddy::xtame::XTAMEDialect, arith::ArithDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<buddy::xtame::XTAMEDialect, arith::ArithDialect,
                           memref::MemRefDialect, scf::SCFDialect>();
    target.addIllegalOp<linalg::MatmulOp, linalg::BatchMatmulOp,
                        linalg::QuantizedMatmulOp>();

    RewritePatternSet patterns(context);
    patterns.add<MatMulToXTAMELowering, BatchMatmulToXTAMELowering>(context);
    patterns.add<QuantizedMatmulToXTAMELowering>(context, maxTemporaryBytes);

    // Check the stack budget before converting, so the diagnostic names the
    // limit instead of the generic legalisation failure.
    bool overBudget = false;
    module.walk([&](linalg::QuantizedMatmulOp op) {
      Value a = op.getDpsInputOperand(0)->get();
      Value b = op.getDpsInputOperand(1)->get();
      Value c = op.getDpsInitOperand(0)->get();
      int64_t m, k, n;
      if (!matmulShape(a, b, c, m, k, n))
        return;
      const int64_t bytes = correctionBytes(m, k, n);
      if (bytes <= maxTemporaryBytes)
        return;
      overBudget = true;
      op.emitError() << "the zero-point correction needs " << bytes
                     << " bytes of stack temporaries, above the "
                     << maxTemporaryBytes
                     << " byte limit (raise --max-temporary-bytes)";
    });
    if (overBudget) {
      signalPassFailure();
      return;
    }

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      module.walk([](Operation *op) {
        if (isa<linalg::MatmulOp, linalg::BatchMatmulOp,
                linalg::QuantizedMatmulOp>(op))
          op->emitError()
              << "unsupported matrix multiply: expected static "
                 "identity-layout i8 operands with an i32 accumulator and at "
                 "least four rows";
      });
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerLowerLinalgToXTAMEPass() {
  PassRegistration<LowerLinalgToXTAMEPass>();
}
} // namespace buddy
} // namespace mlir
