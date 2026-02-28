//====- LowerLinalgToIME.cpp - Linalg to IME Dialect Lowering Pass --------===//
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
// This file defines Linalg dialect lowering pass to IME dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialect/IME/IMEDialect.h"
#include "Dialect/IME/IMEOps.h"

using namespace mlir;
using namespace buddy::ime;


static void getTileSizes(Type elemType, int64_t &tileM, int64_t &tileK,
                         int64_t &tileN) {
  if (elemType.isInteger(8)) {
    tileM = 4;
    tileK = 8;
    tileN = 4;
  } else if (elemType.isInteger(16)) {
    tileM = 4;
    tileK = 4;
    tileN = 4;
  } else {
    // Default to int8 tile sizes
    tileM = 4;
    tileK = 8;
    tileN = 4;
  }
}

static bool isSupportedElementType(Type elemType) {
  return elemType.isInteger(8) || elemType.isInteger(16);
}

namespace {

class MatmulToIMELowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = matmulOp.getLoc();

    Value A = matmulOp.getInputs()[0];  // M x K
    Value B = matmulOp.getInputs()[1];  // K x N
    Value C = matmulOp.getOutputs()[0]; // M x N

    auto AType = dyn_cast<MemRefType>(A.getType());
    auto BType = dyn_cast<MemRefType>(B.getType());
    auto CType = dyn_cast<MemRefType>(C.getType());

    if (!AType || !BType || !CType) {
      return rewriter.notifyMatchFailure(matmulOp,
                                         "operands must be memref types");
    }

    Type AElemType = AType.getElementType();
    Type BElemType = BType.getElementType();
    Type CElemType = CType.getElementType();

    if (!isSupportedElementType(AElemType) ||
        !isSupportedElementType(BElemType)) {
      return rewriter.notifyMatchFailure(
          matmulOp, "only int8 and int16 element types are supported");
    }

    if (AElemType != BElemType) {
      return rewriter.notifyMatchFailure(
          matmulOp, "A and B must have the same element type");
    }

    if (!CElemType.isInteger(32)) {
      return rewriter.notifyMatchFailure(
          matmulOp, "output C must be int32 for accumulation");
    }

    ArrayRef<int64_t> AShape = AType.getShape();
    ArrayRef<int64_t> BShape = BType.getShape();

    if (AShape.size() != 2 || BShape.size() != 2) {
      return rewriter.notifyMatchFailure(matmulOp,
                                         "only 2D matrices are supported");
    }

    int64_t M = AShape[0];
    int64_t K = AShape[1];
    int64_t N = BShape[1];

    bool isDynamic = ShapedType::isDynamic(M) || ShapedType::isDynamic(K) ||
                     ShapedType::isDynamic(N);

    int64_t tileM, tileK, tileN;
    getTileSizes(AElemType, tileM, tileK, tileN);

    // This pattern only handles aligned dimensions.
    // Non-aligned dimensions are handled by MatmulWithBoundaryToIMELowering.
    if (!isDynamic) {
      bool isAligned = (M % tileM == 0) && (K % tileK == 0) && (N % tileN == 0);
      if (!isAligned) {
        return rewriter.notifyMatchFailure(
            matmulOp, "non-aligned dimensions - use boundary handling pattern");
      }
    }
    getTileSizes(AElemType, tileM, tileK, tileN);

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value stepM = rewriter.create<arith::ConstantIndexOp>(loc, tileM);
    Value stepK = rewriter.create<arith::ConstantIndexOp>(loc, tileK);
    Value stepN = rewriter.create<arith::ConstantIndexOp>(loc, tileN);

    Value boundM, boundK, boundN;
    if (isDynamic) {
      boundM = rewriter.create<memref::DimOp>(loc, A, 0);
      boundK = rewriter.create<memref::DimOp>(loc, A, 1);
      boundN = rewriter.create<memref::DimOp>(loc, B, 1);
    } else {
      boundM = rewriter.create<arith::ConstantIndexOp>(loc, M);
      boundK = rewriter.create<arith::ConstantIndexOp>(loc, K);
      boundN = rewriter.create<arith::ConstantIndexOp>(loc, N);
    }

    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value tileKVal = rewriter.create<arith::ConstantIndexOp>(loc, tileK);
    Value tileNVal = rewriter.create<arith::ConstantIndexOp>(loc, tileN);

    // Allocate contiguous buffer for B tile in column-major pack format
    // BTile[N][K] for column-major packing (IME expects B transposed)
    auto BTileType = MemRefType::get({tileN, tileK}, AElemType);
    Value BTileBuffer = rewriter.create<memref::AllocaOp>(loc, BTileType);

    auto loopI = rewriter.create<scf::ForOp>(loc, c0, boundM, stepM);
    rewriter.setInsertionPointToStart(loopI.getBody());
    Value ivI = loopI.getInductionVar();

    auto loopJ = rewriter.create<scf::ForOp>(loc, c0, boundN, stepN);
    rewriter.setInsertionPointToStart(loopJ.getBody());
    Value ivJ = loopJ.getInductionVar();

    auto loopK = rewriter.create<scf::ForOp>(loc, c0, boundK, stepK);
    rewriter.setInsertionPointToStart(loopK.getBody());
    Value ivK = loopK.getInductionVar();

    // A tile: use SubView (no transpose needed)
    SmallVector<OpFoldResult> aOffsets = {ivI, ivK};
    SmallVector<OpFoldResult> aSizes = {rewriter.getIndexAttr(tileM),
                                        rewriter.getIndexAttr(tileK)};
    SmallVector<OpFoldResult> aStrides = {rewriter.getIndexAttr(1),
                                          rewriter.getIndexAttr(1)};

    Value ATile =
        rewriter.create<memref::SubViewOp>(loc, A, aOffsets, aSizes, aStrides);

    // B tile: copy with transpose (B[k][n] -> BTile[n][k])
    // IME requires B to be in column-major pack format
    auto copyBLoopN = rewriter.create<scf::ForOp>(loc, c0, tileNVal, c1);
    rewriter.setInsertionPointToStart(copyBLoopN.getBody());
    Value bn = copyBLoopN.getInductionVar();

    auto copyBLoopK = rewriter.create<scf::ForOp>(loc, c0, tileKVal, c1);
    rewriter.setInsertionPointToStart(copyBLoopK.getBody());
    Value bk = copyBLoopK.getInductionVar();

    // Global indices in B matrix
    Value globalBk = rewriter.create<arith::AddIOp>(loc, ivK, bk);
    Value globalBn = rewriter.create<arith::AddIOp>(loc, ivJ, bn);

    // Load B[globalBk][globalBn] and store to BTile[bn][bk] (transposed)
    Value bVal = rewriter.create<memref::LoadOp>(loc, B, ValueRange{globalBk, globalBn});
    rewriter.create<memref::StoreOp>(loc, bVal, BTileBuffer, ValueRange{bn, bk});

    rewriter.setInsertionPointAfter(copyBLoopN);

    // C tile: use SubView
    SmallVector<OpFoldResult> cOffsets = {ivI, ivJ};
    SmallVector<OpFoldResult> cSizes = {rewriter.getIndexAttr(tileM),
                                        rewriter.getIndexAttr(tileN)};
    SmallVector<OpFoldResult> cStrides = {rewriter.getIndexAttr(1),
                                          rewriter.getIndexAttr(1)};

    Value CTile =
        rewriter.create<memref::SubViewOp>(loc, C, cOffsets, cSizes, cStrides);

    rewriter.create<VmadotOp>(loc, CTile, ATile, BTileBuffer);

    rewriter.setInsertionPointAfter(loopI);

    rewriter.eraseOp(matmulOp);

    return success();
  }
};

class MatmulWithBoundaryToIMELowering
    : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  MatmulWithBoundaryToIMELowering(MLIRContext *context)
      : OpRewritePattern<linalg::MatmulOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = matmulOp.getLoc();

    Value A = matmulOp.getInputs()[0];  // M x K
    Value B = matmulOp.getInputs()[1];  // K x N
    Value C = matmulOp.getOutputs()[0]; // M x N

    auto AType = dyn_cast<MemRefType>(A.getType());
    auto BType = dyn_cast<MemRefType>(B.getType());
    auto CType = dyn_cast<MemRefType>(C.getType());

    if (!AType || !BType || !CType) {
      return rewriter.notifyMatchFailure(matmulOp,
                                         "operands must be memref types");
    }

    Type AElemType = AType.getElementType();
    Type BElemType = BType.getElementType();
    Type CElemType = CType.getElementType();

    if (!isSupportedElementType(AElemType) ||
        !isSupportedElementType(BElemType)) {
      return rewriter.notifyMatchFailure(
          matmulOp, "only int8 and int16 element types are supported");
    }

    if (AElemType != BElemType) {
      return rewriter.notifyMatchFailure(
          matmulOp, "A and B must have the same element type");
    }

    if (!CElemType.isInteger(32)) {
      return rewriter.notifyMatchFailure(
          matmulOp, "output C must be int32 for accumulation");
    }

    ArrayRef<int64_t> AShape = AType.getShape();
    ArrayRef<int64_t> BShape = BType.getShape();

    if (AShape.size() != 2 || BShape.size() != 2) {
      return rewriter.notifyMatchFailure(matmulOp,
                                         "only 2D matrices are supported");
    }

    int64_t M = AShape[0];
    int64_t K = AShape[1];
    int64_t N = BShape[1];

    // Get tile sizes for the element type
    int64_t tileM, tileK, tileN;
    getTileSizes(AElemType, tileM, tileK, tileN);

    // This pattern only handles static dimensions
    bool hasStaticDims = !ShapedType::isDynamic(M) &&
                         !ShapedType::isDynamic(K) && !ShapedType::isDynamic(N);

    if (!hasStaticDims) {
      return rewriter.notifyMatchFailure(
          matmulOp, "dynamic dimensions not supported in boundary pattern");
    }

    // For static dimensions, check if they are aligned
    bool isAligned = (M % tileM == 0) && (K % tileK == 0) && (N % tileN == 0);
    if (isAligned) {
      // Let the simpler MatmulToIMELowering handle aligned cases
      return rewriter.notifyMatchFailure(
          matmulOp, "aligned dimensions - use simple lowering");
    }

    // Calculate number of tiles (ceiling division)
    int64_t numTilesM = (M + tileM - 1) / tileM;
    int64_t numTilesK = (K + tileK - 1) / tileK;
    int64_t numTilesN = (N + tileN - 1) / tileN;

    // Create constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value tileMVal = rewriter.create<arith::ConstantIndexOp>(loc, tileM);
    Value tileKVal = rewriter.create<arith::ConstantIndexOp>(loc, tileK);
    Value tileNVal = rewriter.create<arith::ConstantIndexOp>(loc, tileN);
    Value boundM = rewriter.create<arith::ConstantIndexOp>(loc, M);
    Value boundK = rewriter.create<arith::ConstantIndexOp>(loc, K);
    Value boundN = rewriter.create<arith::ConstantIndexOp>(loc, N);
    Value numTilesMVal = rewriter.create<arith::ConstantIndexOp>(loc, numTilesM);
    Value numTilesKVal = rewriter.create<arith::ConstantIndexOp>(loc, numTilesK);
    Value numTilesNVal = rewriter.create<arith::ConstantIndexOp>(loc, numTilesN);

    // Create zero constants for padding
    Value zeroElem = rewriter.create<arith::ConstantOp>(
        loc, AElemType, rewriter.getZeroAttr(AElemType));
    Value zeroI32 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));

    auto ATileType = MemRefType::get({tileM, tileK}, AElemType);
    auto BTileType = MemRefType::get({tileN, tileK}, AElemType);  // [N, K] for column-major pack
    auto CTileType = MemRefType::get({tileM, tileN}, CElemType);

    Value ATile = rewriter.create<memref::AllocaOp>(loc, ATileType);
    Value BTile = rewriter.create<memref::AllocaOp>(loc, BTileType);
    Value CTile = rewriter.create<memref::AllocaOp>(loc, CTileType);

    // Loop over M tiles
    auto loopTileM = rewriter.create<scf::ForOp>(loc, c0, numTilesMVal, c1);
    rewriter.setInsertionPointToStart(loopTileM.getBody());
    Value tileIdxM = loopTileM.getInductionVar();
    Value baseM = rewriter.create<arith::MulIOp>(loc, tileIdxM, tileMVal);

    // Loop over N tiles
    auto loopTileN = rewriter.create<scf::ForOp>(loc, c0, numTilesNVal, c1);
    rewriter.setInsertionPointToStart(loopTileN.getBody());
    Value tileIdxN = loopTileN.getInductionVar();
    Value baseN = rewriter.create<arith::MulIOp>(loc, tileIdxN, tileNVal);

    // Initialize CTile to zeros (or copy from C for initial values)
    auto initCLoop1 = rewriter.create<scf::ForOp>(loc, c0, tileMVal, c1);
    rewriter.setInsertionPointToStart(initCLoop1.getBody());
    Value initCi = initCLoop1.getInductionVar();
    auto initCLoop2 = rewriter.create<scf::ForOp>(loc, c0, tileNVal, c1);
    rewriter.setInsertionPointToStart(initCLoop2.getBody());
    Value initCj = initCLoop2.getInductionVar();

    // Calculate global indices
    Value globalCi = rewriter.create<arith::AddIOp>(loc, baseM, initCi);
    Value globalCj = rewriter.create<arith::AddIOp>(loc, baseN, initCj);

    // Check if within bounds
    Value inBoundM = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalCi, boundM);
    Value inBoundN = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalCj, boundN);
    Value inBound = rewriter.create<arith::AndIOp>(loc, inBoundM, inBoundN);

    // Load from C if in bounds, else use zero
    auto selectC = rewriter.create<scf::IfOp>(
        loc, CElemType, inBound, /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&selectC.getThenRegion().front());
    Value cLoadVal = rewriter.create<memref::LoadOp>(loc, C, ValueRange{globalCi, globalCj});
    rewriter.create<scf::YieldOp>(loc, cLoadVal);
    rewriter.setInsertionPointToStart(&selectC.getElseRegion().front());
    rewriter.create<scf::YieldOp>(loc, zeroI32);
    rewriter.setInsertionPointAfter(selectC);

    rewriter.create<memref::StoreOp>(loc, selectC.getResult(0), CTile, ValueRange{initCi, initCj});
    rewriter.setInsertionPointAfter(initCLoop1);

    // Loop over K tiles
    auto loopTileK = rewriter.create<scf::ForOp>(loc, c0, numTilesKVal, c1);
    rewriter.setInsertionPointToStart(loopTileK.getBody());
    Value tileIdxK = loopTileK.getInductionVar();
    Value baseK = rewriter.create<arith::MulIOp>(loc, tileIdxK, tileKVal);

    // Copy A tile with boundary handling
    auto copyALoop1 = rewriter.create<scf::ForOp>(loc, c0, tileMVal, c1);
    rewriter.setInsertionPointToStart(copyALoop1.getBody());
    Value copyAi = copyALoop1.getInductionVar();
    auto copyALoop2 = rewriter.create<scf::ForOp>(loc, c0, tileKVal, c1);
    rewriter.setInsertionPointToStart(copyALoop2.getBody());
    Value copyAk = copyALoop2.getInductionVar();

    Value globalAi = rewriter.create<arith::AddIOp>(loc, baseM, copyAi);
    Value globalAk = rewriter.create<arith::AddIOp>(loc, baseK, copyAk);
    Value inBoundAM = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalAi, boundM);
    Value inBoundAK = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalAk, boundK);
    Value inBoundA = rewriter.create<arith::AndIOp>(loc, inBoundAM, inBoundAK);

    auto selectA = rewriter.create<scf::IfOp>(
        loc, AElemType, inBoundA, /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&selectA.getThenRegion().front());
    Value aLoadVal = rewriter.create<memref::LoadOp>(loc, A, ValueRange{globalAi, globalAk});
    rewriter.create<scf::YieldOp>(loc, aLoadVal);
    rewriter.setInsertionPointToStart(&selectA.getElseRegion().front());
    rewriter.create<scf::YieldOp>(loc, zeroElem);
    rewriter.setInsertionPointAfter(selectA);

    rewriter.create<memref::StoreOp>(loc, selectA.getResult(0), ATile, ValueRange{copyAi, copyAk});
    rewriter.setInsertionPointAfter(copyALoop1);

    // Copy B tile with boundary handling
    // Note: B is stored in column-major pack format for IME
    // B[k][n] in original matrix -> BTile[n][k] in packed format
    auto copyBLoop1 = rewriter.create<scf::ForOp>(loc, c0, tileNVal, c1);
    rewriter.setInsertionPointToStart(copyBLoop1.getBody());
    Value copyBn = copyBLoop1.getInductionVar();
    auto copyBLoop2 = rewriter.create<scf::ForOp>(loc, c0, tileKVal, c1);
    rewriter.setInsertionPointToStart(copyBLoop2.getBody());
    Value copyBk = copyBLoop2.getInductionVar();

    Value globalBk = rewriter.create<arith::AddIOp>(loc, baseK, copyBk);
    Value globalBn = rewriter.create<arith::AddIOp>(loc, baseN, copyBn);
    Value inBoundBK = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalBk, boundK);
    Value inBoundBN = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalBn, boundN);
    Value inBoundB = rewriter.create<arith::AndIOp>(loc, inBoundBK, inBoundBN);

    auto selectB = rewriter.create<scf::IfOp>(
        loc, AElemType, inBoundB, /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&selectB.getThenRegion().front());
    // Load from B[k][n] in row-major
    Value bLoadVal = rewriter.create<memref::LoadOp>(loc, B, ValueRange{globalBk, globalBn});
    rewriter.create<scf::YieldOp>(loc, bLoadVal);
    rewriter.setInsertionPointToStart(&selectB.getElseRegion().front());
    rewriter.create<scf::YieldOp>(loc, zeroElem);
    rewriter.setInsertionPointAfter(selectB);

    // Store to BTile[n][k] in column-major pack format
    rewriter.create<memref::StoreOp>(loc, selectB.getResult(0), BTile, ValueRange{copyBn, copyBk});
    rewriter.setInsertionPointAfter(copyBLoop1);

    // IME vmadot on contiguous tile buffers
    rewriter.create<VmadotOp>(loc, CTile, ATile, BTile);

    // End of K tile loop
    rewriter.setInsertionPointAfter(loopTileK);

    // Copy CTile back to C (only valid elements)
    auto storeCLoop1 = rewriter.create<scf::ForOp>(loc, c0, tileMVal, c1);
    rewriter.setInsertionPointToStart(storeCLoop1.getBody());
    Value storeCi = storeCLoop1.getInductionVar();
    auto storeCLoop2 = rewriter.create<scf::ForOp>(loc, c0, tileNVal, c1);
    rewriter.setInsertionPointToStart(storeCLoop2.getBody());
    Value storeCj = storeCLoop2.getInductionVar();

    Value globalStoreCi = rewriter.create<arith::AddIOp>(loc, baseM, storeCi);
    Value globalStoreCj = rewriter.create<arith::AddIOp>(loc, baseN, storeCj);
    Value inBoundStoreM = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalStoreCi, boundM);
    Value inBoundStoreN = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalStoreCj, boundN);
    Value inBoundStore = rewriter.create<arith::AndIOp>(loc, inBoundStoreM, inBoundStoreN);

    auto storeIf = rewriter.create<scf::IfOp>(loc, inBoundStore, /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&storeIf.getThenRegion().front());
    Value cResult = rewriter.create<memref::LoadOp>(loc, CTile, ValueRange{storeCi, storeCj});
    rewriter.create<memref::StoreOp>(loc, cResult, C, ValueRange{globalStoreCi, globalStoreCj});

    rewriter.setInsertionPointAfter(storeCLoop1);

    // End of N and M tile loops
    rewriter.setInsertionPointAfter(loopTileM);

    // Erase the original operation
    rewriter.eraseOp(matmulOp);

    return success();
  }
};

class GenericMatmulToIMELowering : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {

    if (genericOp.getInputs().size() != 2 || genericOp.getOutputs().size() != 1)
      return failure();

    auto iteratorTypes = genericOp.getIteratorTypesArray();
    if (iteratorTypes.size() != 3)
      return failure();

    if (iteratorTypes[0] != utils::IteratorType::parallel ||
        iteratorTypes[1] != utils::IteratorType::parallel ||
        iteratorTypes[2] != utils::IteratorType::reduction)
      return failure();

    auto indexingMaps = genericOp.getIndexingMapsArray();
    if (indexingMaps.size() != 3)
      return failure();

    AffineMap mapA = indexingMaps[0];
    AffineMap mapB = indexingMaps[1];
    AffineMap mapC = indexingMaps[2];

    auto d0 = rewriter.getAffineDimExpr(0);
    auto d1 = rewriter.getAffineDimExpr(1);
    auto d2 = rewriter.getAffineDimExpr(2);

    auto expectedMapA = AffineMap::get(3, 0, {d0, d2}, rewriter.getContext());
    auto expectedMapB = AffineMap::get(3, 0, {d2, d1}, rewriter.getContext());
    auto expectedMapC = AffineMap::get(3, 0, {d0, d1}, rewriter.getContext());

    if (mapA != expectedMapA || mapB != expectedMapB || mapC != expectedMapC)
      return failure();

    Block &body = genericOp.getRegion().front();
    if (body.getOperations().size() != 3)
      return failure();

    auto ops = body.getOperations().begin();
    auto *firstOp = &*ops++;
    auto *secondOp = &*ops++;
    auto *yieldOp = &*ops;

    if (!isa<arith::MulIOp, arith::MulFOp>(firstOp))
      return failure();
    if (!isa<arith::AddIOp, arith::AddFOp>(secondOp))
      return failure();
    if (!isa<linalg::YieldOp>(yieldOp))
      return failure();

    Location loc = genericOp.getLoc();
    Value A = genericOp.getInputs()[0];
    Value B = genericOp.getInputs()[1];
    Value C = genericOp.getOutputs()[0];

    auto AType = dyn_cast<MemRefType>(A.getType());
    auto BType = dyn_cast<MemRefType>(B.getType());
    auto CType = dyn_cast<MemRefType>(C.getType());

    if (!AType || !BType || !CType)
      return failure();

    Type AElemType = AType.getElementType();
    if (!isSupportedElementType(AElemType))
      return failure();

    int64_t tileM, tileK, tileN;
    getTileSizes(AElemType, tileM, tileK, tileN);

    ArrayRef<int64_t> AShape = AType.getShape();
    ArrayRef<int64_t> BShape = BType.getShape();
    int64_t M = AShape[0];
    int64_t K = AShape[1];
    int64_t N = BShape[1];

    bool isDynamic = ShapedType::isDynamic(M) || ShapedType::isDynamic(K) ||
                     ShapedType::isDynamic(N);

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value stepM = rewriter.create<arith::ConstantIndexOp>(loc, tileM);
    Value stepK = rewriter.create<arith::ConstantIndexOp>(loc, tileK);
    Value stepN = rewriter.create<arith::ConstantIndexOp>(loc, tileN);

    Value boundM, boundK, boundN;
    if (isDynamic) {
      boundM = rewriter.create<memref::DimOp>(loc, A, 0);
      boundK = rewriter.create<memref::DimOp>(loc, A, 1);
      boundN = rewriter.create<memref::DimOp>(loc, B, 1);
    } else {
      boundM = rewriter.create<arith::ConstantIndexOp>(loc, M);
      boundK = rewriter.create<arith::ConstantIndexOp>(loc, K);
      boundN = rewriter.create<arith::ConstantIndexOp>(loc, N);
    }

    auto loopI = rewriter.create<scf::ForOp>(loc, c0, boundM, stepM);
    rewriter.setInsertionPointToStart(loopI.getBody());
    Value ivI = loopI.getInductionVar();

    auto loopJ = rewriter.create<scf::ForOp>(loc, c0, boundN, stepN);
    rewriter.setInsertionPointToStart(loopJ.getBody());
    Value ivJ = loopJ.getInductionVar();

    auto loopK = rewriter.create<scf::ForOp>(loc, c0, boundK, stepK);
    rewriter.setInsertionPointToStart(loopK.getBody());
    Value ivK = loopK.getInductionVar();

    SmallVector<OpFoldResult> aOffsets = {ivI, ivK};
    SmallVector<OpFoldResult> aSizes = {rewriter.getIndexAttr(tileM),
                                        rewriter.getIndexAttr(tileK)};
    SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1),
                                         rewriter.getIndexAttr(1)};

    Value ATile =
        rewriter.create<memref::SubViewOp>(loc, A, aOffsets, aSizes, strides);

    SmallVector<OpFoldResult> bOffsets = {ivK, ivJ};
    SmallVector<OpFoldResult> bSizes = {rewriter.getIndexAttr(tileK),
                                        rewriter.getIndexAttr(tileN)};
    Value BTile =
        rewriter.create<memref::SubViewOp>(loc, B, bOffsets, bSizes, strides);

    SmallVector<OpFoldResult> cOffsets = {ivI, ivJ};
    SmallVector<OpFoldResult> cSizes = {rewriter.getIndexAttr(tileM),
                                        rewriter.getIndexAttr(tileN)};
    Value CTile =
        rewriter.create<memref::SubViewOp>(loc, C, cOffsets, cSizes, strides);

    rewriter.create<VmadotOp>(loc, CTile, ATile, BTile);

    rewriter.setInsertionPointAfter(loopI);
    rewriter.eraseOp(genericOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Conv2D to IME Lowering Pattern (with Sliding-Window Instructions)
//===----------------------------------------------------------------------===//

/// Pattern to lower linalg.conv_2d_nhwc_hwcf to IME sliding-window operations.

class Conv2DNhwcHwcfToIMELowering
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
public:
  using OpRewritePattern<linalg::Conv2DNhwcHwcfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    Location loc = convOp.getLoc();

    Value input = convOp.getInputs()[0];
    Value filter = convOp.getInputs()[1];
    Value output = convOp.getOutputs()[0];

    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto filterType = dyn_cast<MemRefType>(filter.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());

    if (!inputType || !filterType || !outputType)
      return rewriter.notifyMatchFailure(convOp,
                                         "operands must be memref types");

    Type inputElemType = inputType.getElementType();
    if (!inputElemType.isInteger(8))
      return rewriter.notifyMatchFailure(convOp, "only int8 is supported");

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> filterShape = filterType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();

    if (inputShape.size() != 4 || filterShape.size() != 4 ||
        outputShape.size() != 4)
      return rewriter.notifyMatchFailure(convOp,
                                         "only 4D tensors are supported");

    int64_t N = inputShape[0];
    int64_t IC = inputShape[3];
    int64_t FH = filterShape[0];
    int64_t FW = filterShape[1];
    int64_t OC = filterShape[3];
    int64_t OH = outputShape[1];
    int64_t OW = outputShape[2];

    int64_t strideH = 1, strideW = 1;

    if (auto stridesAttr = convOp.getStrides()) {
      auto strides = stridesAttr.getValues<int64_t>();
      strideH = strides[0];
      strideW = strides[1];
    }

    const int64_t TILE_M = 4;
    const int64_t TILE_2M = 8;
    const int64_t TILE_K = 8;
    const int64_t TILE_N = 4;

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    Value boundN = rewriter.create<arith::ConstantIndexOp>(loc, N);
    Value boundOH = rewriter.create<arith::ConstantIndexOp>(loc, OH);
    Value boundOW = rewriter.create<arith::ConstantIndexOp>(loc, OW);
    Value boundOC = rewriter.create<arith::ConstantIndexOp>(loc, OC);
    Value boundFH = rewriter.create<arith::ConstantIndexOp>(loc, FH);
    Value boundFW = rewriter.create<arith::ConstantIndexOp>(loc, FW);
    Value boundIC = rewriter.create<arith::ConstantIndexOp>(loc, IC);

    Value stepM = rewriter.create<arith::ConstantIndexOp>(loc, TILE_M);
    Value stepK = rewriter.create<arith::ConstantIndexOp>(loc, TILE_K);
    Value stepN = rewriter.create<arith::ConstantIndexOp>(loc, TILE_N);

    Value strideHVal = rewriter.create<arith::ConstantIndexOp>(loc, strideH);
    Value strideWVal = rewriter.create<arith::ConstantIndexOp>(loc, strideW);

    auto inputTileType = MemRefType::get({TILE_2M, TILE_K}, inputElemType);
    auto filterTileType = MemRefType::get({TILE_K, TILE_N}, inputElemType);
    auto outputTileType =
        MemRefType::get({TILE_M, TILE_N}, rewriter.getI32Type());

    auto loopN = rewriter.create<scf::ForOp>(loc, c0, boundN, c1);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    auto loopOH = rewriter.create<scf::ForOp>(loc, c0, boundOH, stepM);
    rewriter.setInsertionPointToStart(loopOH.getBody());
    Value ivOH = loopOH.getInductionVar();

    auto loopOC = rewriter.create<scf::ForOp>(loc, c0, boundOC, stepN);
    rewriter.setInsertionPointToStart(loopOC.getBody());
    Value ivOC = loopOC.getInductionVar();

    auto loopOW = rewriter.create<scf::ForOp>(loc, c0, boundOW, c1);
    rewriter.setInsertionPointToStart(loopOW.getBody());
    Value ivOW = loopOW.getInductionVar();

    auto loopFH = rewriter.create<scf::ForOp>(loc, c0, boundFH, c1);
    rewriter.setInsertionPointToStart(loopFH.getBody());
    Value ivFH = loopFH.getInductionVar();

    auto loopFW = rewriter.create<scf::ForOp>(loc, c0, boundFW, c1);
    rewriter.setInsertionPointToStart(loopFW.getBody());
    Value ivFW = loopFW.getInductionVar();

    auto loopIC = rewriter.create<scf::ForOp>(loc, c0, boundIC, stepK);
    rewriter.setInsertionPointToStart(loopIC.getBody());
    Value ivIC = loopIC.getInductionVar();

    Value inputTile = rewriter.create<memref::AllocaOp>(loc, inputTileType);
    Value filterTile = rewriter.create<memref::AllocaOp>(loc, filterTileType);
    Value outputTile = rewriter.create<memref::AllocaOp>(loc, outputTileType);

    Value ihBase = rewriter.create<arith::MulIOp>(loc, ivOH, strideHVal);
    ihBase = rewriter.create<arith::AddIOp>(loc, ihBase, ivFH);
    Value iw = rewriter.create<arith::MulIOp>(loc, ivOW, strideWVal);
    iw = rewriter.create<arith::AddIOp>(loc, iw, ivFW);

    Value tileMBound = rewriter.create<arith::ConstantIndexOp>(loc, TILE_M);
    Value tile2MBound = rewriter.create<arith::ConstantIndexOp>(loc, TILE_2M);
    Value tileKBound = rewriter.create<arith::ConstantIndexOp>(loc, TILE_K);
    Value tileNBound = rewriter.create<arith::ConstantIndexOp>(loc, TILE_N);

    auto fillInputLoop = rewriter.create<scf::ForOp>(loc, c0, tile2MBound, c1);
    rewriter.setInsertionPointToStart(fillInputLoop.getBody());
    Value fillM = fillInputLoop.getInductionVar();

    auto fillInputInnerLoop =
        rewriter.create<scf::ForOp>(loc, c0, tileKBound, c1);
    rewriter.setInsertionPointToStart(fillInputInnerLoop.getBody());
    Value fillK = fillInputInnerLoop.getInductionVar();

    Value mTimesStride = rewriter.create<arith::MulIOp>(loc, fillM, strideHVal);
    Value inputIH = rewriter.create<arith::AddIOp>(loc, ihBase, mTimesStride);
    Value inputIC = rewriter.create<arith::AddIOp>(loc, ivIC, fillK);
    Value inputVal = rewriter.create<memref::LoadOp>(
        loc, input, ValueRange{ivN, inputIH, iw, inputIC});
    rewriter.create<memref::StoreOp>(loc, inputVal, inputTile,
                                     ValueRange{fillM, fillK});

    rewriter.setInsertionPointAfter(fillInputLoop);

    auto fillFilterLoop = rewriter.create<scf::ForOp>(loc, c0, tileKBound, c1);
    rewriter.setInsertionPointToStart(fillFilterLoop.getBody());
    Value fillFK = fillFilterLoop.getInductionVar();

    auto fillFilterInnerLoop =
        rewriter.create<scf::ForOp>(loc, c0, tileNBound, c1);
    rewriter.setInsertionPointToStart(fillFilterInnerLoop.getBody());
    Value fillFN = fillFilterInnerLoop.getInductionVar();

    Value filterIC = rewriter.create<arith::AddIOp>(loc, ivIC, fillFK);
    Value filterOC = rewriter.create<arith::AddIOp>(loc, ivOC, fillFN);
    Value filterVal = rewriter.create<memref::LoadOp>(
        loc, filter, ValueRange{ivFH, ivFW, filterIC, filterOC});
    rewriter.create<memref::StoreOp>(loc, filterVal, filterTile,
                                     ValueRange{fillFK, fillFN});

    rewriter.setInsertionPointAfter(fillFilterLoop);

    auto loadOutputLoop = rewriter.create<scf::ForOp>(loc, c0, tileMBound, c1);
    rewriter.setInsertionPointToStart(loadOutputLoop.getBody());
    Value loadM = loadOutputLoop.getInductionVar();

    auto loadOutputInnerLoop =
        rewriter.create<scf::ForOp>(loc, c0, tileNBound, c1);
    rewriter.setInsertionPointToStart(loadOutputInnerLoop.getBody());
    Value loadN = loadOutputInnerLoop.getInductionVar();

    Value outOH = rewriter.create<arith::AddIOp>(loc, ivOH, loadM);
    Value outOC = rewriter.create<arith::AddIOp>(loc, ivOC, loadN);
    Value outVal = rewriter.create<memref::LoadOp>(
        loc, output, ValueRange{ivN, outOH, ivOW, outOC});
    rewriter.create<memref::StoreOp>(loc, outVal, outputTile,
                                     ValueRange{loadM, loadN});

    rewriter.setInsertionPointAfter(loadOutputLoop);

    if (strideH == 1) {
      rewriter.create<Vmadot1Op>(loc, outputTile, inputTile, filterTile);
    } else if (strideH == 2) {
      rewriter.create<Vmadot2Op>(loc, outputTile, inputTile, filterTile);
    } else if (strideH == 3) {
      rewriter.create<Vmadot3Op>(loc, outputTile, inputTile, filterTile);
    } else {
      Value slide = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
      rewriter.create<VmadotnOp>(loc, outputTile, inputTile, filterTile, slide);
    }

    auto storeOutputLoop = rewriter.create<scf::ForOp>(loc, c0, tileMBound, c1);
    rewriter.setInsertionPointToStart(storeOutputLoop.getBody());
    Value storeM = storeOutputLoop.getInductionVar();

    auto storeOutputInnerLoop =
        rewriter.create<scf::ForOp>(loc, c0, tileNBound, c1);
    rewriter.setInsertionPointToStart(storeOutputInnerLoop.getBody());
    Value storeN = storeOutputInnerLoop.getInductionVar();

    Value storeOH = rewriter.create<arith::AddIOp>(loc, ivOH, storeM);
    Value storeOC = rewriter.create<arith::AddIOp>(loc, ivOC, storeN);
    Value storeVal = rewriter.create<memref::LoadOp>(
        loc, outputTile, ValueRange{storeM, storeN});
    rewriter.create<memref::StoreOp>(loc, storeVal, output,
                                     ValueRange{ivN, storeOH, ivOW, storeOC});

    rewriter.setInsertionPointAfter(storeOutputLoop);

    rewriter.setInsertionPointAfter(loopN);

    rewriter.eraseOp(convOp);

    return success();
  }
};

/// Pattern to lower linalg.conv_2d_nchw_fchw to IME sliding-window operations.

class Conv2DNchwFchwToIMELowering
    : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
public:
  using OpRewritePattern<linalg::Conv2DNchwFchwOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter &rewriter) const override {
    Location loc = convOp.getLoc();

    Value input = convOp.getInputs()[0];
    Value filter = convOp.getInputs()[1];
    Value output = convOp.getOutputs()[0];

    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto filterType = dyn_cast<MemRefType>(filter.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());

    if (!inputType || !filterType || !outputType)
      return rewriter.notifyMatchFailure(convOp,
                                         "operands must be memref types");

    Type inputElemType = inputType.getElementType();
    if (!inputElemType.isInteger(8))
      return rewriter.notifyMatchFailure(convOp, "only int8 is supported");

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> filterShape = filterType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();

    if (inputShape.size() != 4 || filterShape.size() != 4 ||
        outputShape.size() != 4)
      return rewriter.notifyMatchFailure(convOp,
                                         "only 4D tensors are supported");

    int64_t N = inputShape[0];
    int64_t IC = inputShape[1];
    int64_t OC = filterShape[0];
    int64_t FH = filterShape[2];
    int64_t FW = filterShape[3];
    int64_t OH = outputShape[2];
    int64_t OW = outputShape[3];

    int64_t strideH = 1, strideW = 1;

    if (auto stridesAttr = convOp.getStrides()) {
      auto strides = stridesAttr.getValues<int64_t>();
      strideH = strides[0];
      strideW = strides[1];
    }

    const int64_t TILE_M = 4;
    const int64_t TILE_2M = 8;
    const int64_t TILE_K = 8;
    const int64_t TILE_N = 4;

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    Value boundN = rewriter.create<arith::ConstantIndexOp>(loc, N);
    Value boundOC = rewriter.create<arith::ConstantIndexOp>(loc, OC);
    Value boundOH = rewriter.create<arith::ConstantIndexOp>(loc, OH);
    Value boundOW = rewriter.create<arith::ConstantIndexOp>(loc, OW);
    Value boundIC = rewriter.create<arith::ConstantIndexOp>(loc, IC);
    Value boundFH = rewriter.create<arith::ConstantIndexOp>(loc, FH);
    Value boundFW = rewriter.create<arith::ConstantIndexOp>(loc, FW);

    Value stepM = rewriter.create<arith::ConstantIndexOp>(loc, TILE_M);
    Value stepK = rewriter.create<arith::ConstantIndexOp>(loc, TILE_K);
    Value stepN = rewriter.create<arith::ConstantIndexOp>(loc, TILE_N);

    Value strideHVal = rewriter.create<arith::ConstantIndexOp>(loc, strideH);
    Value strideWVal = rewriter.create<arith::ConstantIndexOp>(loc, strideW);

    auto loopN = rewriter.create<scf::ForOp>(loc, c0, boundN, c1);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    auto loopOC = rewriter.create<scf::ForOp>(loc, c0, boundOC, stepN);
    rewriter.setInsertionPointToStart(loopOC.getBody());
    Value ivOC = loopOC.getInductionVar();

    auto loopOH = rewriter.create<scf::ForOp>(loc, c0, boundOH, stepM);
    rewriter.setInsertionPointToStart(loopOH.getBody());
    Value ivOH = loopOH.getInductionVar();

    auto loopOW = rewriter.create<scf::ForOp>(loc, c0, boundOW, c1);
    rewriter.setInsertionPointToStart(loopOW.getBody());
    Value ivOW = loopOW.getInductionVar();

    auto loopIC = rewriter.create<scf::ForOp>(loc, c0, boundIC, stepK);
    rewriter.setInsertionPointToStart(loopIC.getBody());
    Value ivIC = loopIC.getInductionVar();

    auto loopFH = rewriter.create<scf::ForOp>(loc, c0, boundFH, c1);
    rewriter.setInsertionPointToStart(loopFH.getBody());
    Value ivFH = loopFH.getInductionVar();

    auto loopFW = rewriter.create<scf::ForOp>(loc, c0, boundFW, c1);
    rewriter.setInsertionPointToStart(loopFW.getBody());
    Value ivFW = loopFW.getInductionVar();

    Value ihBase = rewriter.create<arith::MulIOp>(loc, ivOH, strideHVal);
    ihBase = rewriter.create<arith::AddIOp>(loc, ihBase, ivFH);
    Value iw = rewriter.create<arith::MulIOp>(loc, ivOW, strideWVal);
    iw = rewriter.create<arith::AddIOp>(loc, iw, ivFW);

    auto inputTileType = MemRefType::get({TILE_2M, TILE_K}, inputElemType);
    auto filterTileType = MemRefType::get({TILE_K, TILE_N}, inputElemType);
    auto outputTileType =
        MemRefType::get({TILE_M, TILE_N}, rewriter.getI32Type());

    Value inputTile = rewriter.create<memref::AllocaOp>(loc, inputTileType);
    Value filterTile = rewriter.create<memref::AllocaOp>(loc, filterTileType);
    Value outputTile = rewriter.create<memref::AllocaOp>(loc, outputTileType);

    Value tileMBound = rewriter.create<arith::ConstantIndexOp>(loc, TILE_M);
    Value tile2MBound = rewriter.create<arith::ConstantIndexOp>(loc, TILE_2M);
    Value tileKBound = rewriter.create<arith::ConstantIndexOp>(loc, TILE_K);
    Value tileNBound = rewriter.create<arith::ConstantIndexOp>(loc, TILE_N);

    auto fillInputLoop = rewriter.create<scf::ForOp>(loc, c0, tile2MBound, c1);
    rewriter.setInsertionPointToStart(fillInputLoop.getBody());
    Value fillM = fillInputLoop.getInductionVar();

    auto fillInputInnerLoop =
        rewriter.create<scf::ForOp>(loc, c0, tileKBound, c1);
    rewriter.setInsertionPointToStart(fillInputInnerLoop.getBody());
    Value fillK = fillInputInnerLoop.getInductionVar();

    Value mTimesStride = rewriter.create<arith::MulIOp>(loc, fillM, strideHVal);
    Value inputIH = rewriter.create<arith::AddIOp>(loc, ihBase, mTimesStride);
    Value inputIC = rewriter.create<arith::AddIOp>(loc, ivIC, fillK);
    Value inputVal = rewriter.create<memref::LoadOp>(
        loc, input, ValueRange{ivN, inputIC, inputIH, iw});
    rewriter.create<memref::StoreOp>(loc, inputVal, inputTile,
                                     ValueRange{fillM, fillK});

    rewriter.setInsertionPointAfter(fillInputLoop);

    auto fillFilterLoop = rewriter.create<scf::ForOp>(loc, c0, tileKBound, c1);
    rewriter.setInsertionPointToStart(fillFilterLoop.getBody());
    Value fillFK = fillFilterLoop.getInductionVar();

    auto fillFilterInnerLoop =
        rewriter.create<scf::ForOp>(loc, c0, tileNBound, c1);
    rewriter.setInsertionPointToStart(fillFilterInnerLoop.getBody());
    Value fillFN = fillFilterInnerLoop.getInductionVar();

    Value filterOC = rewriter.create<arith::AddIOp>(loc, ivOC, fillFN);
    Value filterIC = rewriter.create<arith::AddIOp>(loc, ivIC, fillFK);
    Value filterVal = rewriter.create<memref::LoadOp>(
        loc, filter, ValueRange{filterOC, filterIC, ivFH, ivFW});
    rewriter.create<memref::StoreOp>(loc, filterVal, filterTile,
                                     ValueRange{fillFK, fillFN});

    rewriter.setInsertionPointAfter(fillFilterLoop);

    auto loadOutputLoop = rewriter.create<scf::ForOp>(loc, c0, tileMBound, c1);
    rewriter.setInsertionPointToStart(loadOutputLoop.getBody());
    Value loadM = loadOutputLoop.getInductionVar();

    auto loadOutputInnerLoop =
        rewriter.create<scf::ForOp>(loc, c0, tileNBound, c1);
    rewriter.setInsertionPointToStart(loadOutputInnerLoop.getBody());
    Value loadN = loadOutputInnerLoop.getInductionVar();

    Value outOC = rewriter.create<arith::AddIOp>(loc, ivOC, loadN);
    Value outOH = rewriter.create<arith::AddIOp>(loc, ivOH, loadM);
    Value outVal = rewriter.create<memref::LoadOp>(
        loc, output, ValueRange{ivN, outOC, outOH, ivOW});
    rewriter.create<memref::StoreOp>(loc, outVal, outputTile,
                                     ValueRange{loadM, loadN});

    rewriter.setInsertionPointAfter(loadOutputLoop);

    if (strideH == 1) {
      rewriter.create<Vmadot1Op>(loc, outputTile, inputTile, filterTile);
    } else if (strideH == 2) {
      rewriter.create<Vmadot2Op>(loc, outputTile, inputTile, filterTile);
    } else if (strideH == 3) {
      rewriter.create<Vmadot3Op>(loc, outputTile, inputTile, filterTile);
    } else {
      Value slide = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
      rewriter.create<VmadotnOp>(loc, outputTile, inputTile, filterTile, slide);
    }

    auto storeOutputLoop = rewriter.create<scf::ForOp>(loc, c0, tileMBound, c1);
    rewriter.setInsertionPointToStart(storeOutputLoop.getBody());
    Value storeM = storeOutputLoop.getInductionVar();

    auto storeOutputInnerLoop =
        rewriter.create<scf::ForOp>(loc, c0, tileNBound, c1);
    rewriter.setInsertionPointToStart(storeOutputInnerLoop.getBody());
    Value storeN = storeOutputInnerLoop.getInductionVar();

    Value storeOC = rewriter.create<arith::AddIOp>(loc, ivOC, storeN);
    Value storeOH = rewriter.create<arith::AddIOp>(loc, ivOH, storeM);
    Value storeVal = rewriter.create<memref::LoadOp>(
        loc, outputTile, ValueRange{storeM, storeN});
    rewriter.create<memref::StoreOp>(loc, storeVal, output,
                                     ValueRange{ivN, storeOC, storeOH, ivOW});

    rewriter.setInsertionPointAfter(storeOutputLoop);

    rewriter.setInsertionPointAfter(loopN);

    rewriter.eraseOp(convOp);

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
class LowerLinalgToIMEPass
    : public PassWrapper<LowerLinalgToIMEPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToIMEPass)

  StringRef getArgument() const final { return "lower-linalg-to-ime"; }
  StringRef getDescription() const final {
    return "Lower linalg dialect operations to IME dialect operations.";
  }

  LowerLinalgToIMEPass() = default;
  LowerLinalgToIMEPass(const LowerLinalgToIMEPass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IMEDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LowerLinalgToIMEPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  RewritePatternSet patterns(context);

  // Add patterns with higher benefit first (aligned dimensions)
  patterns.add<MatmulToIMELowering>(context);
  patterns.add<GenericMatmulToIMELowering>(context);

  // Add boundary handling pattern with lower benefit (tried after aligned cases fail)
  patterns.add<MatmulWithBoundaryToIMELowering>(context);

  patterns.add<Conv2DNhwcHwcfToIMELowering>(context);
  patterns.add<Conv2DNchwFchwToIMELowering>(context);

  if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace buddy {
void registerLowerLinalgToIMEPass() {
  PassRegistration<LowerLinalgToIMEPass>();
}
} // namespace buddy
} // namespace mlir
