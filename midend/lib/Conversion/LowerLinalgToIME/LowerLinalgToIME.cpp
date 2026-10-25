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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialect/IME/IMEDialect.h"
#include "Dialect/IME/IMEOps.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace buddy::ime;

static void getTileSizes(Type elemType, int64_t &tileM, int64_t &tileK,
                         int64_t &tileN, bool k3 = false) {
  if (k3) {
    tileM = tileN = 8;
    tileK = elemType.isF16() ? 8 : 16;
    return;
  }
  if (elemType.isInteger(8)) {
    tileM = 4;
    tileK = 8;
    tileN = 4;
  } else if (elemType.isInteger(16)) {
    tileM = 4;
    tileK = 4;
    tileN = 4;
  } else if (elemType.isF16()) {
    // FP16: MAC unit 4x4x4
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
  return elemType.isInteger(8) || elemType.isInteger(16) || elemType.isF16();
}

namespace {

// Unsupported floating-point IME encodings can behave as a no-op on some
// hardware. Check a known nonzero product before enabling a model's IME path.
static void insertFP16IMECheck(PatternRewriter &rewriter, Location loc,
                               ModuleOp module, bool a100,
                               StringRef functionName) {
  // C-wrapper export makes helper symbols external. Scope each helper to its
  // graph so separately compiled Prefill and Decode partitions can be linked.
  std::string name =
      (a100 ? "__buddy_check_a100_fp16_ime_" : "__buddy_check_fp16_ime_") +
      functionName.str();
  if (!module.lookupSymbol<func::FuncOp>(name)) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto function = func::FuncOp::create(rewriter, loc, name,
                                         rewriter.getFunctionType({}, {}));
    function.setPrivate();
    rewriter.setInsertionPointToStart(function.addEntryBlock());
    int64_t size = a100 ? 8 : 4;
    Type accumulatorType = a100 ? rewriter.getF32Type() : rewriter.getF16Type();
    auto tileType = MemRefType::get({size, size}, rewriter.getF16Type());
    Value a = memref::AllocaOp::create(rewriter, loc, tileType);
    Value b = memref::AllocaOp::create(rewriter, loc, tileType);
    Value c = memref::AllocaOp::create(
        rewriter, loc, MemRefType::get({size, size}, accumulatorType));
    Value zero =
        arith::ConstantOp::create(rewriter, loc, accumulatorType,
                                  rewriter.getFloatAttr(accumulatorType, 0.0));
    Value one = arith::ConstantOp::create(
        rewriter, loc, rewriter.getF16Type(),
        rewriter.getFloatAttr(rewriter.getF16Type(), 1.0));
    Value expected = arith::ConstantOp::create(
        rewriter, loc, accumulatorType,
        rewriter.getFloatAttr(accumulatorType, static_cast<double>(size)));
    for (int64_t i = 0; i < size; ++i)
      for (int64_t j = 0; j < size; ++j) {
        Value row = arith::ConstantIndexOp::create(rewriter, loc, i);
        Value col = arith::ConstantIndexOp::create(rewriter, loc, j);
        memref::StoreOp::create(rewriter, loc, one, a, ValueRange{row, col});
        memref::StoreOp::create(rewriter, loc, one, b, ValueRange{row, col});
        memref::StoreOp::create(rewriter, loc, zero, c, ValueRange{row, col});
      }
    if (a100)
      VfwmadotOp::create(rewriter, loc, c, a, b);
    else
      VfmadotOp::create(rewriter, loc, c, a, b);
    Value indexZero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value actual = memref::LoadOp::create(rewriter, loc, c,
                                          ValueRange{indexZero, indexZero});
    Value valid = arith::CmpFOp::create(
        rewriter, loc, arith::CmpFPredicate::OEQ, actual, expected);
    cf::AssertOp::create(
        rewriter, loc, valid,
        "FP16 IME numerical self-test failed; this hardware does not execute "
        "the selected matrix instruction correctly");
    func::ReturnOp::create(rewriter, loc);
  }
  func::CallOp::create(rewriter, loc, name, TypeRange{}, ValueRange{});
}

class MatmulToIMELowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  MatmulToIMELowering(MLIRContext *context, bool k3, StringRef shapes = "",
                      int64_t packedWidth = 0, bool parallelTiles = false,
                      bool verifyFP16 = false, bool a100 = false,
                      bool fastGemv = false, bool fastGemm = false)
      : OpRewritePattern(context), k3(k3), shapes(shapes.str()),
        packedWidth(packedWidth), parallelTiles(parallelTiles),
        verifyFP16(verifyFP16), a100(a100), fastGemv(fastGemv),
        fastGemm(fastGemm) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = matmulOp.getLoc();

    if (k3) {
      auto maps = matmulOp.getIndexingMapsArray();
      auto m = rewriter.getAffineDimExpr(0);
      auto n = rewriter.getAffineDimExpr(1);
      auto k = rewriter.getAffineDimExpr(2);
      auto ctx = rewriter.getContext();
      if (maps != SmallVector<AffineMap>{AffineMap::get(3, 0, {m, k}, ctx),
                                         AffineMap::get(3, 0, {k, n}, ctx),
                                         AffineMap::get(3, 0, {m, n}, ctx)})
        return rewriter.notifyMatchFailure(
            matmulOp, "requires standard matmul indexing maps");
    }

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
        !isSupportedElementType(BElemType) ||
        (k3 && (AElemType.isInteger(16) || BElemType.isInteger(16)))) {
      return rewriter.notifyMatchFailure(
          matmulOp, "only int8, int16, and f16 element types are supported");
    }

    if (AElemType != BElemType) {
      return rewriter.notifyMatchFailure(
          matmulOp, "A and B must have the same element type");
    }

    bool isFloatMatmul = AElemType.isF16();
    if (isFloatMatmul) {
      if (!CElemType.isF16() && !(a100 && CElemType.isF32())) {
        return rewriter.notifyMatchFailure(
            matmulOp, "output C must be f16 for fp16 accumulation");
      }
    } else {
      if (!CElemType.isInteger(32)) {
        return rewriter.notifyMatchFailure(
            matmulOp, "output C must be int32 for integer accumulation");
      }
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
    getTileSizes(AElemType, tileM, tileK, tileN, k3);
    bool wideningFP16 = a100 && isFloatMatmul;
    if (wideningFP16)
      tileM = tileK = tileN = 8;
    Type accumulatorType = wideningFP16 ? rewriter.getF32Type() : CElemType;

    // This pattern only handles static dimensions
    bool hasStaticDims = !ShapedType::isDynamic(M) &&
                         !ShapedType::isDynamic(K) && !ShapedType::isDynamic(N);

    if (!hasStaticDims) {
      return rewriter.notifyMatchFailure(
          matmulOp, "dynamic dimensions not supported in boundary pattern");
    }

    if (!shapes.empty()) {
      SmallVector<StringRef> selected;
      StringRef(shapes).split(selected, ',', -1, false);
      std::string shape = std::to_string(K) + "x" + std::to_string(N);
      if (!llvm::is_contained(selected, StringRef(shape)))
        return failure();
    }
    if (packedWidth && (M != 1 || N % packedWidth != 0))
      return rewriter.notifyMatchFailure(
          matmulOp, "packed IME requires M=1 and N divisible by packed width");
    if (fastGemm &&
        (!a100 || !isFloatMatmul || !CElemType.isF16() || M < 8 || M % 8 ||
         K % 8 || N % 32 || !AType.getLayout().isIdentity() ||
         !CType.getLayout().isIdentity() || packedWidth))
      return rewriter.notifyMatchFailure(
          matmulOp, "A100 GEMM requires contiguous FP16 A/C, M%8=K%8=N%32=0");
    if (verifyFP16 && isFloatMatmul)
      insertFP16IMECheck(
          rewriter, loc, matmulOp->getParentOfType<ModuleOp>(), a100,
          matmulOp->getParentOfType<func::FuncOp>().getSymName());

    if (fastGemm) {
      auto index = [&](int64_t value) -> Value {
        return arith::ConstantIndexOp::create(rewriter, loc, value);
      };
      Value zero = index(0), eight = index(8), one = index(1);
      Value boundM = index(M), boundK = index(K), boundN = index(N);
      Value packedA = memref::AllocOp::create(
          rewriter, loc, MemRefType::get({M * K}, AElemType));
      auto packRows =
          scf::ParallelOp::create(rewriter, loc, ValueRange{zero},
                                  ValueRange{boundM}, ValueRange{eight});
      rewriter.setInsertionPointToStart(packRows.getBody());
      Value row = packRows.getInductionVars()[0];
      Value rowBase = arith::MulIOp::create(rewriter, loc, row, boundK);
      auto packK = scf::ForOp::create(rewriter, loc, zero, boundK, eight);
      rewriter.setInsertionPointToStart(packK.getBody());
      Value kb = packK.getInductionVar();
      Value block = arith::MulIOp::create(rewriter, loc, kb, eight);
      block = arith::AddIOp::create(rewriter, loc, block, rowBase);
      auto packRow = scf::ForOp::create(rewriter, loc, zero, eight, one);
      rewriter.setInsertionPointToStart(packRow.getBody());
      Value r = packRow.getInductionVar();
      Value sourceRow = arith::AddIOp::create(rewriter, loc, row, r);
      Value offset = arith::MulIOp::create(rewriter, loc, r, eight);
      offset = arith::AddIOp::create(rewriter, loc, offset, block);
      Value vector =
          vector::LoadOp::create(rewriter, loc, VectorType::get({8}, AElemType),
                                 A, ValueRange{sourceRow, kb});
      vector::StoreOp::create(rewriter, loc, vector, packedA,
                              ValueRange{offset});
      rewriter.setInsertionPointAfter(packRows);

      auto columns =
          scf::ParallelOp::create(rewriter, loc, ValueRange{zero},
                                  ValueRange{boundN}, ValueRange{index(32)});
      rewriter.setInsertionPointToStart(columns.getBody());
      Value column = columns.getInductionVars()[0];
      Value packedB = memref::AllocOp::create(
          rewriter, loc, MemRefType::get({K * 32}, BElemType));
      PackBF16Op::create(rewriter, loc, packedB, B, column,
                         rewriter.getI64IntegerAttr(K));
      auto rows = scf::ForOp::create(rewriter, loc, zero, boundM, eight);
      rewriter.setInsertionPointToStart(rows.getBody());
      GemmF16Op::create(
          rewriter, loc, C, packedA, packedB, rows.getInductionVar(), column,
          rewriter.getI64IntegerAttr(K), rewriter.getI64IntegerAttr(N));
      rewriter.setInsertionPointAfter(rows);
      memref::DeallocOp::create(rewriter, loc, packedB);
      rewriter.setInsertionPointAfter(columns);
      memref::DeallocOp::create(rewriter, loc, packedA);
      rewriter.eraseOp(matmulOp);
      return success();
    }

    if (fastGemv) {
      if (!a100 || !isFloatMatmul || !CElemType.isF16() || M != 1 ||
          packedWidth != 32 || K % 16 || N % 32)
        return matmulOp.emitError("A100 fast GEMV requires FP16 M=1, K%16=0, "
                                  "N%32=0 and tile32-packed weights");
      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value bound = arith::ConstantIndexOp::create(rewriter, loc, N);
      Value step = arith::ConstantIndexOp::create(rewriter, loc, 32);
      bool unroll = N <= 256 || K == N;
      int64_t prefetch = N <= 256 ? 0 : (unroll ? 1024 : 512);
      auto loop = scf::ParallelOp::create(rewriter, loc, ValueRange{zero},
                                          ValueRange{bound}, ValueRange{step});
      rewriter.setInsertionPointToStart(loop.getBody());
      GemvF16Op::create(rewriter, loc, C, A, B, loop.getInductionVars()[0],
                        rewriter.getI64IntegerAttr(K),
                        rewriter.getI64IntegerAttr(prefetch),
                        rewriter.getBoolAttr(unroll));
      rewriter.eraseOp(matmulOp);
      return success();
    }

    // Calculate number of tiles (ceiling division)
    int64_t numTilesM = (M + tileM - 1) / tileM;
    int64_t numTilesK = (K + tileK - 1) / tileK;
    int64_t numTilesN = (N + tileN - 1) / tileN;

    // Create constants
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value tileMVal = arith::ConstantIndexOp::create(rewriter, loc, tileM);
    Value tileKVal = arith::ConstantIndexOp::create(rewriter, loc, tileK);
    Value tileNVal = arith::ConstantIndexOp::create(rewriter, loc, tileN);
    Value boundM = arith::ConstantIndexOp::create(rewriter, loc, M);
    Value boundK = arith::ConstantIndexOp::create(rewriter, loc, K);
    Value boundN = arith::ConstantIndexOp::create(rewriter, loc, N);
    Value numTilesMVal =
        arith::ConstantIndexOp::create(rewriter, loc, numTilesM);
    Value numTilesKVal =
        arith::ConstantIndexOp::create(rewriter, loc, numTilesK);
    Value numTilesNVal =
        arith::ConstantIndexOp::create(rewriter, loc, numTilesN);

    Value flatB;
    Value packedWidthVal;
    if (packedWidth) {
      auto metadata =
          memref::ExtractStridedMetadataOp::create(rewriter, loc, B);
      auto layout = StridedLayoutAttr::get(
          rewriter.getContext(), ShapedType::kDynamic, ArrayRef<int64_t>{1});
      auto flatType = MemRefType::get({K * N}, BElemType, layout);
      flatB = memref::ReinterpretCastOp::create(
          rewriter, loc, flatType, metadata.getBaseBuffer(),
          metadata.getOffset(),
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(K * N)},
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)});
      packedWidthVal =
          arith::ConstantIndexOp::create(rewriter, loc, packedWidth);
    }

    // Create zero constants for padding
    Value zeroElem = arith::ConstantOp::create(rewriter, loc, AElemType,
                                               rewriter.getZeroAttr(AElemType));
    // Zero constant for C tile (type depends on element type: f16 or i32)
    Value zeroC = arith::ConstantOp::create(
        rewriter, loc, accumulatorType, rewriter.getZeroAttr(accumulatorType));

    auto ATileType = MemRefType::get({tileM, tileK}, AElemType);
    auto BTileType = MemRefType::get({tileN, tileK},
                                     AElemType); // [N, K] for column-major pack
    auto CTileType = MemRefType::get({tileM, tileN}, accumulatorType);

    Value ATile, BTile, CTile;
    if (!parallelTiles) {
      ATile = memref::AllocaOp::create(rewriter, loc, ATileType);
      BTile = memref::AllocaOp::create(rewriter, loc, BTileType);
      CTile = memref::AllocaOp::create(rewriter, loc, CTileType);
    }

    // Loop over M tiles
    auto loopTileM = scf::ForOp::create(rewriter, loc, c0, numTilesMVal, c1);
    rewriter.setInsertionPointToStart(loopTileM.getBody());
    Value tileIdxM = loopTileM.getInductionVar();
    Value baseM = arith::MulIOp::create(rewriter, loc, tileIdxM, tileMVal);

    // Loop over N tiles
    Value tileIdxN;
    if (parallelTiles) {
      auto loopTileN =
          scf::ParallelOp::create(rewriter, loc, ValueRange{c0},
                                  ValueRange{numTilesNVal}, ValueRange{c1});
      rewriter.setInsertionPointToStart(loopTileN.getBody());
      tileIdxN = loopTileN.getInductionVars()[0];
    } else {
      auto loopTileN = scf::ForOp::create(rewriter, loc, c0, numTilesNVal, c1);
      rewriter.setInsertionPointToStart(loopTileN.getBody());
      tileIdxN = loopTileN.getInductionVar();
    }
    Value baseN = arith::MulIOp::create(rewriter, loc, tileIdxN, tileNVal);

    // Each parallel N tile needs its own scratch matrices.
    if (parallelTiles) {
      ATile = memref::AllocaOp::create(rewriter, loc, ATileType);
      BTile = memref::AllocaOp::create(rewriter, loc, BTileType);
      CTile = memref::AllocaOp::create(rewriter, loc, CTileType);
    }

    // Initialize CTile to zeros (or copy from C for initial values)
    auto initCLoop1 = scf::ForOp::create(rewriter, loc, c0, tileMVal, c1);
    rewriter.setInsertionPointToStart(initCLoop1.getBody());
    Value initCi = initCLoop1.getInductionVar();
    auto initCLoop2 = scf::ForOp::create(rewriter, loc, c0, tileNVal, c1);
    rewriter.setInsertionPointToStart(initCLoop2.getBody());
    Value initCj = initCLoop2.getInductionVar();

    // Calculate global indices
    Value globalCi = arith::AddIOp::create(rewriter, loc, baseM, initCi);
    Value globalCj = arith::AddIOp::create(rewriter, loc, baseN, initCj);

    // Check if within bounds
    Value inBoundM = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, globalCi, boundM);
    Value inBoundN = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, globalCj, boundN);
    Value inBound = arith::AndIOp::create(rewriter, loc, inBoundM, inBoundN);

    // Load from C if in bounds, else use zero
    auto selectC = scf::IfOp::create(rewriter, loc, accumulatorType, inBound,
                                     /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&selectC.getThenRegion().front());
    Value cLoadVal = memref::LoadOp::create(rewriter, loc, C,
                                            ValueRange{globalCi, globalCj});
    if (wideningFP16 && CElemType.isF16())
      cLoadVal =
          arith::ExtFOp::create(rewriter, loc, accumulatorType, cLoadVal);
    scf::YieldOp::create(rewriter, loc, cLoadVal);
    rewriter.setInsertionPointToStart(&selectC.getElseRegion().front());
    scf::YieldOp::create(rewriter, loc, zeroC);
    rewriter.setInsertionPointAfter(selectC);

    memref::StoreOp::create(rewriter, loc, selectC.getResult(0), CTile,
                            ValueRange{initCi, initCj});
    rewriter.setInsertionPointAfter(initCLoop1);

    // Loop over K tiles
    auto loopTileK = scf::ForOp::create(rewriter, loc, c0, numTilesKVal, c1);
    rewriter.setInsertionPointToStart(loopTileK.getBody());
    Value tileIdxK = loopTileK.getInductionVar();
    Value baseK = arith::MulIOp::create(rewriter, loc, tileIdxK, tileKVal);

    // Copy A tile with boundary handling
    auto copyALoop1 = scf::ForOp::create(rewriter, loc, c0, tileMVal, c1);
    rewriter.setInsertionPointToStart(copyALoop1.getBody());
    Value copyAi = copyALoop1.getInductionVar();
    auto copyALoop2 = scf::ForOp::create(rewriter, loc, c0, tileKVal, c1);
    rewriter.setInsertionPointToStart(copyALoop2.getBody());
    Value copyAk = copyALoop2.getInductionVar();

    Value globalAi = arith::AddIOp::create(rewriter, loc, baseM, copyAi);
    Value globalAk = arith::AddIOp::create(rewriter, loc, baseK, copyAk);
    Value inBoundAM = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, globalAi, boundM);
    Value inBoundAK = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, globalAk, boundK);
    Value inBoundA = arith::AndIOp::create(rewriter, loc, inBoundAM, inBoundAK);

    auto selectA = scf::IfOp::create(rewriter, loc, AElemType, inBoundA,
                                     /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&selectA.getThenRegion().front());
    Value aLoadVal = memref::LoadOp::create(rewriter, loc, A,
                                            ValueRange{globalAi, globalAk});
    scf::YieldOp::create(rewriter, loc, aLoadVal);
    rewriter.setInsertionPointToStart(&selectA.getElseRegion().front());
    scf::YieldOp::create(rewriter, loc, zeroElem);
    rewriter.setInsertionPointAfter(selectA);

    memref::StoreOp::create(rewriter, loc, selectA.getResult(0), ATile,
                            ValueRange{copyAi, copyAk});
    rewriter.setInsertionPointAfter(copyALoop1);

    // Copy B tile with boundary handling
    // Note: B is stored in column-major pack format for IME
    // B[k][n] in original matrix -> BTile[n][k] in packed format
    auto copyBLoop1 = scf::ForOp::create(rewriter, loc, c0, tileNVal, c1);
    rewriter.setInsertionPointToStart(copyBLoop1.getBody());
    Value copyBn = copyBLoop1.getInductionVar();
    auto copyBLoop2 = scf::ForOp::create(rewriter, loc, c0, tileKVal, c1);
    rewriter.setInsertionPointToStart(copyBLoop2.getBody());
    Value copyBk = copyBLoop2.getInductionVar();

    Value globalBk = arith::AddIOp::create(rewriter, loc, baseK, copyBk);
    Value globalBn = arith::AddIOp::create(rewriter, loc, baseN, copyBn);
    Value inBoundBK = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, globalBk, boundK);
    Value inBoundBN = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, globalBn, boundN);
    Value inBoundB = arith::AndIOp::create(rewriter, loc, inBoundBK, inBoundBN);

    auto selectB = scf::IfOp::create(rewriter, loc, AElemType, inBoundB,
                                     /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&selectB.getThenRegion().front());
    Value bLoadVal;
    if (packedWidth) {
      Value panel =
          arith::DivUIOp::create(rewriter, loc, globalBn, packedWidthVal);
      Value lane =
          arith::RemUIOp::create(rewriter, loc, globalBn, packedWidthVal);
      Value panelBase = arith::MulIOp::create(rewriter, loc, panel, boundK);
      Value row = arith::AddIOp::create(rewriter, loc, panelBase, globalBk);
      Value offset = arith::MulIOp::create(rewriter, loc, row, packedWidthVal);
      offset = arith::AddIOp::create(rewriter, loc, offset, lane);
      bLoadVal =
          memref::LoadOp::create(rewriter, loc, flatB, ValueRange{offset});
    } else {
      bLoadVal = memref::LoadOp::create(rewriter, loc, B,
                                        ValueRange{globalBk, globalBn});
    }
    scf::YieldOp::create(rewriter, loc, bLoadVal);
    rewriter.setInsertionPointToStart(&selectB.getElseRegion().front());
    scf::YieldOp::create(rewriter, loc, zeroElem);
    rewriter.setInsertionPointAfter(selectB);

    // Store to BTile[n][k] in column-major pack format
    memref::StoreOp::create(rewriter, loc, selectB.getResult(0), BTile,
                            ValueRange{copyBn, copyBk});
    rewriter.setInsertionPointAfter(copyBLoop1);

    // IME vmadot/vfmadot on contiguous tile buffers
    if (wideningFP16) {
      VfwmadotOp::create(rewriter, loc, CTile, ATile, BTile);
    } else if (AElemType.isF16()) {
      VfmadotOp::create(rewriter, loc, CTile, ATile, BTile);
    } else {
      VmadotOp::create(rewriter, loc, CTile, ATile, BTile);
    }

    // End of K tile loop
    rewriter.setInsertionPointAfter(loopTileK);

    // Copy CTile back to C (only valid elements)
    auto storeCLoop1 = scf::ForOp::create(rewriter, loc, c0, tileMVal, c1);
    rewriter.setInsertionPointToStart(storeCLoop1.getBody());
    Value storeCi = storeCLoop1.getInductionVar();
    auto storeCLoop2 = scf::ForOp::create(rewriter, loc, c0, tileNVal, c1);
    rewriter.setInsertionPointToStart(storeCLoop2.getBody());
    Value storeCj = storeCLoop2.getInductionVar();

    Value globalStoreCi = arith::AddIOp::create(rewriter, loc, baseM, storeCi);
    Value globalStoreCj = arith::AddIOp::create(rewriter, loc, baseN, storeCj);
    Value inBoundStoreM = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, globalStoreCi, boundM);
    Value inBoundStoreN = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, globalStoreCj, boundN);
    Value inBoundStore =
        arith::AndIOp::create(rewriter, loc, inBoundStoreM, inBoundStoreN);

    auto storeIf = scf::IfOp::create(rewriter, loc, inBoundStore,
                                     /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&storeIf.getThenRegion().front());
    Value cResult = memref::LoadOp::create(rewriter, loc, CTile,
                                           ValueRange{storeCi, storeCj});
    if (wideningFP16 && CElemType.isF16())
      cResult = arith::TruncFOp::create(rewriter, loc, CElemType, cResult);
    memref::StoreOp::create(rewriter, loc, cResult, C,
                            ValueRange{globalStoreCi, globalStoreCj});

    rewriter.setInsertionPointAfter(storeCLoop1);

    // End of N and M tile loops
    rewriter.setInsertionPointAfter(loopTileM);

    // Erase the original operation
    rewriter.eraseOp(matmulOp);

    return success();
  }

private:
  bool k3;
  std::string shapes;
  int64_t packedWidth;
  bool parallelTiles;
  bool verifyFP16;
  bool a100;
  bool fastGemv;
  bool fastGemm;
};

class GenericMatmulToIMELowering : public OpRewritePattern<linalg::GenericOp> {
public:
  GenericMatmulToIMELowering(MLIRContext *context, bool k3)
      : OpRewritePattern<linalg::GenericOp>(context), k3(k3) {}

private:
  bool k3;

public:
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

    if (k3) {
      // Use the boundary-aware named matmul path, including B transposition.
      // A generic with the same op names but a different scalar computation
      // must not be mistaken for a dot product.
      auto args = body.getArguments();
      auto samePair = [](Value a, Value b, Value x, Value y) {
        return (a == x && b == y) || (a == y && b == x);
      };
      if (AElemType.isInteger(16) ||
          !samePair(firstOp->getOperand(0), firstOp->getOperand(1), args[0],
                    args[1]) ||
          !samePair(secondOp->getOperand(0), secondOp->getOperand(1),
                    firstOp->getResult(0), args[2]) ||
          yieldOp->getOperand(0) != secondOp->getResult(0))
        return failure();
      linalg::MatmulOp::create(rewriter, loc, ValueRange{A, B}, ValueRange{C});
      rewriter.eraseOp(genericOp);
      return success();
    }

    bool isFloatGeneric = AElemType.isF16();
    int64_t tileM, tileK, tileN;
    getTileSizes(AElemType, tileM, tileK, tileN, k3);

    ArrayRef<int64_t> AShape = AType.getShape();
    ArrayRef<int64_t> BShape = BType.getShape();
    int64_t M = AShape[0];
    int64_t K = AShape[1];
    int64_t N = BShape[1];

    bool isDynamic = ShapedType::isDynamic(M) || ShapedType::isDynamic(K) ||
                     ShapedType::isDynamic(N);

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value stepM = arith::ConstantIndexOp::create(rewriter, loc, tileM);
    Value stepK = arith::ConstantIndexOp::create(rewriter, loc, tileK);
    Value stepN = arith::ConstantIndexOp::create(rewriter, loc, tileN);

    Value boundM, boundK, boundN;
    if (isDynamic) {
      boundM = memref::DimOp::create(rewriter, loc, A, 0);
      boundK = memref::DimOp::create(rewriter, loc, A, 1);
      boundN = memref::DimOp::create(rewriter, loc, B, 1);
    } else {
      boundM = arith::ConstantIndexOp::create(rewriter, loc, M);
      boundK = arith::ConstantIndexOp::create(rewriter, loc, K);
      boundN = arith::ConstantIndexOp::create(rewriter, loc, N);
    }

    auto loopI = scf::ForOp::create(rewriter, loc, c0, boundM, stepM);
    rewriter.setInsertionPointToStart(loopI.getBody());
    Value ivI = loopI.getInductionVar();

    auto loopJ = scf::ForOp::create(rewriter, loc, c0, boundN, stepN);
    rewriter.setInsertionPointToStart(loopJ.getBody());
    Value ivJ = loopJ.getInductionVar();

    auto loopK = scf::ForOp::create(rewriter, loc, c0, boundK, stepK);
    rewriter.setInsertionPointToStart(loopK.getBody());
    Value ivK = loopK.getInductionVar();

    SmallVector<OpFoldResult> aOffsets = {ivI, ivK};
    SmallVector<OpFoldResult> aSizes = {rewriter.getIndexAttr(tileM),
                                        rewriter.getIndexAttr(tileK)};
    SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1),
                                         rewriter.getIndexAttr(1)};

    Value ATile =
        memref::SubViewOp::create(rewriter, loc, A, aOffsets, aSizes, strides);

    SmallVector<OpFoldResult> bOffsets = {ivK, ivJ};
    SmallVector<OpFoldResult> bSizes = {rewriter.getIndexAttr(tileK),
                                        rewriter.getIndexAttr(tileN)};
    Value BTile =
        memref::SubViewOp::create(rewriter, loc, B, bOffsets, bSizes, strides);

    SmallVector<OpFoldResult> cOffsets = {ivI, ivJ};
    SmallVector<OpFoldResult> cSizes = {rewriter.getIndexAttr(tileM),
                                        rewriter.getIndexAttr(tileN)};
    Value CTile =
        memref::SubViewOp::create(rewriter, loc, C, cOffsets, cSizes, strides);

    if (isFloatGeneric) {
      VfmadotOp::create(rewriter, loc, CTile, ATile, BTile);
    } else {
      VmadotOp::create(rewriter, loc, CTile, ATile, BTile);
    }

    rewriter.setInsertionPointAfter(loopI);
    rewriter.eraseOp(genericOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Conv2D to IME Lowering Pattern (with Sliding-Window Instructions)
//===----------------------------------------------------------------------===//

/// Pattern to lower linalg.batch_matmul_transpose_b to IME operations.
/// Handles batched matmul where B is transposed: C[b,m,n] += A[b,m,k] *
/// B[b,n,k] B is already in [N,K] layout per batch, which is IME's expected
/// column-major pack format — no repack needed.

class BatchMatmulTransposeBToIMELowering
    : public OpRewritePattern<linalg::BatchMatmulOp> {
public:
  BatchMatmulTransposeBToIMELowering(MLIRContext *context, bool k3)
      : OpRewritePattern<linalg::BatchMatmulOp>(context), k3(k3) {}

private:
  bool k3;

public:
  LogicalResult matchAndRewrite(linalg::BatchMatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (!isa<linalg::BatchMatmulTransposeBOp>(op.getOperation()))
      return rewriter.notifyMatchFailure(op,
                                         "requires transposed B indexing maps");

    Value A = op.getInputs()[0];  // [Batch, M, K]
    Value B = op.getInputs()[1];  // [Batch, N, K] (transposed)
    Value C = op.getOutputs()[0]; // [Batch, M, N]

    auto AType = dyn_cast<MemRefType>(A.getType());
    auto BType = dyn_cast<MemRefType>(B.getType());
    auto CType = dyn_cast<MemRefType>(C.getType());

    if (!AType || !BType || !CType)
      return rewriter.notifyMatchFailure(op, "operands must be memref types");

    Type AElemType = AType.getElementType();
    Type BElemType = BType.getElementType();
    Type CElemType = CType.getElementType();

    if (!isSupportedElementType(AElemType) ||
        !isSupportedElementType(BElemType) ||
        (k3 && (AElemType.isInteger(16) || BElemType.isInteger(16))))
      return rewriter.notifyMatchFailure(
          op, "only int8, int16, and f16 element types are supported");

    if (AElemType != BElemType)
      return rewriter.notifyMatchFailure(
          op, "A and B must have the same element type");

    bool isFloat = AElemType.isF16();
    if (isFloat) {
      if (!CElemType.isF16())
        return rewriter.notifyMatchFailure(
            op, "output C must be f16 for fp16 accumulation");
    } else {
      if (!CElemType.isInteger(32))
        return rewriter.notifyMatchFailure(
            op, "output C must be int32 for integer accumulation");
    }

    ArrayRef<int64_t> AShape = AType.getShape(); // [Batch, M, K]
    ArrayRef<int64_t> BShape = BType.getShape(); // [Batch, N, K]

    if (AShape.size() != 3 || BShape.size() != 3)
      return rewriter.notifyMatchFailure(op, "only 3D tensors are supported");

    int64_t Batch = AShape[0];
    int64_t M = AShape[1];
    int64_t K = AShape[2];
    int64_t N = BShape[1];

    if (ShapedType::isDynamic(Batch) || ShapedType::isDynamic(M) ||
        ShapedType::isDynamic(K) || ShapedType::isDynamic(N))
      return rewriter.notifyMatchFailure(op,
                                         "dynamic dimensions not supported");

    int64_t tileM, tileK, tileN;
    getTileSizes(AElemType, tileM, tileK, tileN, k3);

    int64_t numTilesM = (M + tileM - 1) / tileM;
    int64_t numTilesK = (K + tileK - 1) / tileK;
    int64_t numTilesN = (N + tileN - 1) / tileN;

    // Constants
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value tileMVal = arith::ConstantIndexOp::create(rewriter, loc, tileM);
    Value tileKVal = arith::ConstantIndexOp::create(rewriter, loc, tileK);
    Value tileNVal = arith::ConstantIndexOp::create(rewriter, loc, tileN);
    Value boundM = arith::ConstantIndexOp::create(rewriter, loc, M);
    Value boundK = arith::ConstantIndexOp::create(rewriter, loc, K);
    Value boundN = arith::ConstantIndexOp::create(rewriter, loc, N);
    Value numTilesMVal =
        arith::ConstantIndexOp::create(rewriter, loc, numTilesM);
    Value numTilesKVal =
        arith::ConstantIndexOp::create(rewriter, loc, numTilesK);
    Value numTilesNVal =
        arith::ConstantIndexOp::create(rewriter, loc, numTilesN);
    Value batchBound = arith::ConstantIndexOp::create(rewriter, loc, Batch);

    Value zeroElem = arith::ConstantOp::create(rewriter, loc, AElemType,
                                               rewriter.getZeroAttr(AElemType));
    Value zeroC = arith::ConstantOp::create(rewriter, loc, CElemType,
                                            rewriter.getZeroAttr(CElemType));

    auto ATileType = MemRefType::get({tileM, tileK}, AElemType);
    // B is [N,K] per batch — already column-major pack for IME
    auto BTileType = MemRefType::get({tileN, tileK}, AElemType);
    auto CTileType = MemRefType::get({tileM, tileN}, CElemType);

    Value ATile = memref::AllocaOp::create(rewriter, loc, ATileType);
    Value BTile = memref::AllocaOp::create(rewriter, loc, BTileType);
    Value CTile = memref::AllocaOp::create(rewriter, loc, CTileType);

    // Batch loop
    auto loopBatch = scf::ForOp::create(rewriter, loc, c0, batchBound, c1);
    rewriter.setInsertionPointToStart(loopBatch.getBody());
    Value batchIdx = loopBatch.getInductionVar();

    // Loop over M tiles
    auto loopTileM = scf::ForOp::create(rewriter, loc, c0, numTilesMVal, c1);
    rewriter.setInsertionPointToStart(loopTileM.getBody());
    Value tileIdxM = loopTileM.getInductionVar();
    Value baseM = arith::MulIOp::create(rewriter, loc, tileIdxM, tileMVal);

    // Loop over N tiles
    auto loopTileN = scf::ForOp::create(rewriter, loc, c0, numTilesNVal, c1);
    rewriter.setInsertionPointToStart(loopTileN.getBody());
    Value tileIdxN = loopTileN.getInductionVar();
    Value baseN = arith::MulIOp::create(rewriter, loc, tileIdxN, tileNVal);

    // Initialize CTile from C[batch, m, n]
    auto initCLoop1 = scf::ForOp::create(rewriter, loc, c0, tileMVal, c1);
    rewriter.setInsertionPointToStart(initCLoop1.getBody());
    Value initCi = initCLoop1.getInductionVar();
    auto initCLoop2 = scf::ForOp::create(rewriter, loc, c0, tileNVal, c1);
    rewriter.setInsertionPointToStart(initCLoop2.getBody());
    Value initCj = initCLoop2.getInductionVar();

    Value globalCi = arith::AddIOp::create(rewriter, loc, baseM, initCi);
    Value globalCj = arith::AddIOp::create(rewriter, loc, baseN, initCj);

    Value inBoundM = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, globalCi, boundM);
    Value inBoundN = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, globalCj, boundN);
    Value inBound = arith::AndIOp::create(rewriter, loc, inBoundM, inBoundN);

    auto selectC = scf::IfOp::create(rewriter, loc, CElemType, inBound,
                                     /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&selectC.getThenRegion().front());
    Value cLoadVal = memref::LoadOp::create(
        rewriter, loc, C, ValueRange{batchIdx, globalCi, globalCj});
    scf::YieldOp::create(rewriter, loc, cLoadVal);
    rewriter.setInsertionPointToStart(&selectC.getElseRegion().front());
    scf::YieldOp::create(rewriter, loc, zeroC);
    rewriter.setInsertionPointAfter(selectC);

    memref::StoreOp::create(rewriter, loc, selectC.getResult(0), CTile,
                            ValueRange{initCi, initCj});
    rewriter.setInsertionPointAfter(initCLoop1);

    // K tile loop
    auto loopTileK = scf::ForOp::create(rewriter, loc, c0, numTilesKVal, c1);
    rewriter.setInsertionPointToStart(loopTileK.getBody());
    Value tileIdxK = loopTileK.getInductionVar();
    Value baseK = arith::MulIOp::create(rewriter, loc, tileIdxK, tileKVal);

    // Copy A tile: A[batch, m, k] → ATile[tileM, tileK]
    auto copyALoop1 = scf::ForOp::create(rewriter, loc, c0, tileMVal, c1);
    rewriter.setInsertionPointToStart(copyALoop1.getBody());
    Value copyAi = copyALoop1.getInductionVar();
    auto copyALoop2 = scf::ForOp::create(rewriter, loc, c0, tileKVal, c1);
    rewriter.setInsertionPointToStart(copyALoop2.getBody());
    Value copyAk = copyALoop2.getInductionVar();

    Value globalAi = arith::AddIOp::create(rewriter, loc, baseM, copyAi);
    Value globalAk = arith::AddIOp::create(rewriter, loc, baseK, copyAk);
    Value inBoundAM = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, globalAi, boundM);
    Value inBoundAK = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, globalAk, boundK);
    Value inBoundA = arith::AndIOp::create(rewriter, loc, inBoundAM, inBoundAK);

    auto selectA = scf::IfOp::create(rewriter, loc, AElemType, inBoundA,
                                     /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&selectA.getThenRegion().front());
    Value aLoadVal = memref::LoadOp::create(
        rewriter, loc, A, ValueRange{batchIdx, globalAi, globalAk});
    scf::YieldOp::create(rewriter, loc, aLoadVal);
    rewriter.setInsertionPointToStart(&selectA.getElseRegion().front());
    scf::YieldOp::create(rewriter, loc, zeroElem);
    rewriter.setInsertionPointAfter(selectA);

    memref::StoreOp::create(rewriter, loc, selectA.getResult(0), ATile,
                            ValueRange{copyAi, copyAk});
    rewriter.setInsertionPointAfter(copyALoop1);

    // Copy B tile: B[batch, n, k] → BTile[tileN, tileK]
    // B is already [N,K] per batch — same as IME's expected format!
    // No transpose/repack needed.
    auto copyBLoop1 = scf::ForOp::create(rewriter, loc, c0, tileNVal, c1);
    rewriter.setInsertionPointToStart(copyBLoop1.getBody());
    Value copyBn = copyBLoop1.getInductionVar();
    auto copyBLoop2 = scf::ForOp::create(rewriter, loc, c0, tileKVal, c1);
    rewriter.setInsertionPointToStart(copyBLoop2.getBody());
    Value copyBk = copyBLoop2.getInductionVar();

    Value globalBn = arith::AddIOp::create(rewriter, loc, baseN, copyBn);
    Value globalBk = arith::AddIOp::create(rewriter, loc, baseK, copyBk);
    Value inBoundBN = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, globalBn, boundN);
    Value inBoundBK = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, globalBk, boundK);
    Value inBoundB = arith::AndIOp::create(rewriter, loc, inBoundBN, inBoundBK);

    auto selectB = scf::IfOp::create(rewriter, loc, AElemType, inBoundB,
                                     /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&selectB.getThenRegion().front());
    // B[batch, n, k] — already column-major
    Value bLoadVal = memref::LoadOp::create(
        rewriter, loc, B, ValueRange{batchIdx, globalBn, globalBk});
    scf::YieldOp::create(rewriter, loc, bLoadVal);
    rewriter.setInsertionPointToStart(&selectB.getElseRegion().front());
    scf::YieldOp::create(rewriter, loc, zeroElem);
    rewriter.setInsertionPointAfter(selectB);

    // BTile[n,k] — direct copy, no repack
    memref::StoreOp::create(rewriter, loc, selectB.getResult(0), BTile,
                            ValueRange{copyBn, copyBk});
    rewriter.setInsertionPointAfter(copyBLoop1);

    // IME compute
    if (isFloat) {
      VfmadotOp::create(rewriter, loc, CTile, ATile, BTile);
    } else {
      VmadotOp::create(rewriter, loc, CTile, ATile, BTile);
    }

    // End K tile loop
    rewriter.setInsertionPointAfter(loopTileK);

    // Store CTile back to C[batch, m, n]
    auto storeCLoop1 = scf::ForOp::create(rewriter, loc, c0, tileMVal, c1);
    rewriter.setInsertionPointToStart(storeCLoop1.getBody());
    Value storeCi = storeCLoop1.getInductionVar();
    auto storeCLoop2 = scf::ForOp::create(rewriter, loc, c0, tileNVal, c1);
    rewriter.setInsertionPointToStart(storeCLoop2.getBody());
    Value storeCj = storeCLoop2.getInductionVar();

    Value globalStoreCi = arith::AddIOp::create(rewriter, loc, baseM, storeCi);
    Value globalStoreCj = arith::AddIOp::create(rewriter, loc, baseN, storeCj);
    Value inBoundStoreM = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, globalStoreCi, boundM);
    Value inBoundStoreN = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, globalStoreCj, boundN);
    Value inBoundStore =
        arith::AndIOp::create(rewriter, loc, inBoundStoreM, inBoundStoreN);

    auto storeIf = scf::IfOp::create(rewriter, loc, inBoundStore,
                                     /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&storeIf.getThenRegion().front());
    Value cResult = memref::LoadOp::create(rewriter, loc, CTile,
                                           ValueRange{storeCi, storeCj});
    memref::StoreOp::create(rewriter, loc, cResult, C,
                            ValueRange{batchIdx, globalStoreCi, globalStoreCj});

    rewriter.setInsertionPointAfter(storeCLoop1);

    // End N, M, Batch loops
    rewriter.setInsertionPointAfter(loopBatch);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Conv2D to IME Lowering Patterns (with Sliding-Window Instructions)
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

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

    Value boundN = arith::ConstantIndexOp::create(rewriter, loc, N);
    Value boundOH = arith::ConstantIndexOp::create(rewriter, loc, OH);
    Value boundOW = arith::ConstantIndexOp::create(rewriter, loc, OW);
    Value boundOC = arith::ConstantIndexOp::create(rewriter, loc, OC);
    Value boundFH = arith::ConstantIndexOp::create(rewriter, loc, FH);
    Value boundFW = arith::ConstantIndexOp::create(rewriter, loc, FW);
    Value boundIC = arith::ConstantIndexOp::create(rewriter, loc, IC);

    Value stepM = arith::ConstantIndexOp::create(rewriter, loc, TILE_M);
    Value stepK = arith::ConstantIndexOp::create(rewriter, loc, TILE_K);
    Value stepN = arith::ConstantIndexOp::create(rewriter, loc, TILE_N);

    Value strideHVal = arith::ConstantIndexOp::create(rewriter, loc, strideH);
    Value strideWVal = arith::ConstantIndexOp::create(rewriter, loc, strideW);

    auto inputTileType = MemRefType::get({TILE_2M, TILE_K}, inputElemType);
    auto filterTileType = MemRefType::get({TILE_K, TILE_N}, inputElemType);
    auto outputTileType =
        MemRefType::get({TILE_M, TILE_N}, rewriter.getI32Type());

    auto loopN = scf::ForOp::create(rewriter, loc, c0, boundN, c1);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    auto loopOH = scf::ForOp::create(rewriter, loc, c0, boundOH, stepM);
    rewriter.setInsertionPointToStart(loopOH.getBody());
    Value ivOH = loopOH.getInductionVar();

    auto loopOC = scf::ForOp::create(rewriter, loc, c0, boundOC, stepN);
    rewriter.setInsertionPointToStart(loopOC.getBody());
    Value ivOC = loopOC.getInductionVar();

    auto loopOW = scf::ForOp::create(rewriter, loc, c0, boundOW, c1);
    rewriter.setInsertionPointToStart(loopOW.getBody());
    Value ivOW = loopOW.getInductionVar();

    auto loopFH = scf::ForOp::create(rewriter, loc, c0, boundFH, c1);
    rewriter.setInsertionPointToStart(loopFH.getBody());
    Value ivFH = loopFH.getInductionVar();

    auto loopFW = scf::ForOp::create(rewriter, loc, c0, boundFW, c1);
    rewriter.setInsertionPointToStart(loopFW.getBody());
    Value ivFW = loopFW.getInductionVar();

    auto loopIC = scf::ForOp::create(rewriter, loc, c0, boundIC, stepK);
    rewriter.setInsertionPointToStart(loopIC.getBody());
    Value ivIC = loopIC.getInductionVar();

    Value inputTile = memref::AllocaOp::create(rewriter, loc, inputTileType);
    Value filterTile = memref::AllocaOp::create(rewriter, loc, filterTileType);
    Value outputTile = memref::AllocaOp::create(rewriter, loc, outputTileType);

    Value ihBase = arith::MulIOp::create(rewriter, loc, ivOH, strideHVal);
    ihBase = arith::AddIOp::create(rewriter, loc, ihBase, ivFH);
    Value iw = arith::MulIOp::create(rewriter, loc, ivOW, strideWVal);
    iw = arith::AddIOp::create(rewriter, loc, iw, ivFW);

    Value tileMBound = arith::ConstantIndexOp::create(rewriter, loc, TILE_M);
    Value tile2MBound = arith::ConstantIndexOp::create(rewriter, loc, TILE_2M);
    Value tileKBound = arith::ConstantIndexOp::create(rewriter, loc, TILE_K);
    Value tileNBound = arith::ConstantIndexOp::create(rewriter, loc, TILE_N);

    auto fillInputLoop = scf::ForOp::create(rewriter, loc, c0, tile2MBound, c1);
    rewriter.setInsertionPointToStart(fillInputLoop.getBody());
    Value fillM = fillInputLoop.getInductionVar();

    auto fillInputInnerLoop =
        scf::ForOp::create(rewriter, loc, c0, tileKBound, c1);
    rewriter.setInsertionPointToStart(fillInputInnerLoop.getBody());
    Value fillK = fillInputInnerLoop.getInductionVar();

    Value mTimesStride =
        arith::MulIOp::create(rewriter, loc, fillM, strideHVal);
    Value inputIH = arith::AddIOp::create(rewriter, loc, ihBase, mTimesStride);
    Value inputIC = arith::AddIOp::create(rewriter, loc, ivIC, fillK);
    Value inputVal = memref::LoadOp::create(
        rewriter, loc, input, ValueRange{ivN, inputIH, iw, inputIC});
    memref::StoreOp::create(rewriter, loc, inputVal, inputTile,
                            ValueRange{fillM, fillK});

    rewriter.setInsertionPointAfter(fillInputLoop);

    auto fillFilterLoop = scf::ForOp::create(rewriter, loc, c0, tileKBound, c1);
    rewriter.setInsertionPointToStart(fillFilterLoop.getBody());
    Value fillFK = fillFilterLoop.getInductionVar();

    auto fillFilterInnerLoop =
        scf::ForOp::create(rewriter, loc, c0, tileNBound, c1);
    rewriter.setInsertionPointToStart(fillFilterInnerLoop.getBody());
    Value fillFN = fillFilterInnerLoop.getInductionVar();

    Value filterIC = arith::AddIOp::create(rewriter, loc, ivIC, fillFK);
    Value filterOC = arith::AddIOp::create(rewriter, loc, ivOC, fillFN);
    Value filterVal = memref::LoadOp::create(
        rewriter, loc, filter, ValueRange{ivFH, ivFW, filterIC, filterOC});
    memref::StoreOp::create(rewriter, loc, filterVal, filterTile,
                            ValueRange{fillFK, fillFN});

    rewriter.setInsertionPointAfter(fillFilterLoop);

    auto loadOutputLoop = scf::ForOp::create(rewriter, loc, c0, tileMBound, c1);
    rewriter.setInsertionPointToStart(loadOutputLoop.getBody());
    Value loadM = loadOutputLoop.getInductionVar();

    auto loadOutputInnerLoop =
        scf::ForOp::create(rewriter, loc, c0, tileNBound, c1);
    rewriter.setInsertionPointToStart(loadOutputInnerLoop.getBody());
    Value loadN = loadOutputInnerLoop.getInductionVar();

    Value outOH = arith::AddIOp::create(rewriter, loc, ivOH, loadM);
    Value outOC = arith::AddIOp::create(rewriter, loc, ivOC, loadN);
    Value outVal = memref::LoadOp::create(rewriter, loc, output,
                                          ValueRange{ivN, outOH, ivOW, outOC});
    memref::StoreOp::create(rewriter, loc, outVal, outputTile,
                            ValueRange{loadM, loadN});

    rewriter.setInsertionPointAfter(loadOutputLoop);

    if (strideH == 1) {
      Vmadot1Op::create(rewriter, loc, outputTile, inputTile, filterTile);
    } else if (strideH == 2) {
      Vmadot2Op::create(rewriter, loc, outputTile, inputTile, filterTile);
    } else if (strideH == 3) {
      Vmadot3Op::create(rewriter, loc, outputTile, inputTile, filterTile);
    } else {
      Value slide = arith::ConstantIntOp::create(rewriter, loc, 0, 64);
      VmadotnOp::create(rewriter, loc, outputTile, inputTile, filterTile,
                        slide);
    }

    auto storeOutputLoop =
        scf::ForOp::create(rewriter, loc, c0, tileMBound, c1);
    rewriter.setInsertionPointToStart(storeOutputLoop.getBody());
    Value storeM = storeOutputLoop.getInductionVar();

    auto storeOutputInnerLoop =
        scf::ForOp::create(rewriter, loc, c0, tileNBound, c1);
    rewriter.setInsertionPointToStart(storeOutputInnerLoop.getBody());
    Value storeN = storeOutputInnerLoop.getInductionVar();

    Value storeOH = arith::AddIOp::create(rewriter, loc, ivOH, storeM);
    Value storeOC = arith::AddIOp::create(rewriter, loc, ivOC, storeN);
    Value storeVal = memref::LoadOp::create(rewriter, loc, outputTile,
                                            ValueRange{storeM, storeN});
    memref::StoreOp::create(rewriter, loc, storeVal, output,
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

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

    Value boundN = arith::ConstantIndexOp::create(rewriter, loc, N);
    Value boundOC = arith::ConstantIndexOp::create(rewriter, loc, OC);
    Value boundOH = arith::ConstantIndexOp::create(rewriter, loc, OH);
    Value boundOW = arith::ConstantIndexOp::create(rewriter, loc, OW);
    Value boundIC = arith::ConstantIndexOp::create(rewriter, loc, IC);
    Value boundFH = arith::ConstantIndexOp::create(rewriter, loc, FH);
    Value boundFW = arith::ConstantIndexOp::create(rewriter, loc, FW);

    Value stepM = arith::ConstantIndexOp::create(rewriter, loc, TILE_M);
    Value stepK = arith::ConstantIndexOp::create(rewriter, loc, TILE_K);
    Value stepN = arith::ConstantIndexOp::create(rewriter, loc, TILE_N);

    Value strideHVal = arith::ConstantIndexOp::create(rewriter, loc, strideH);
    Value strideWVal = arith::ConstantIndexOp::create(rewriter, loc, strideW);

    auto loopN = scf::ForOp::create(rewriter, loc, c0, boundN, c1);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    auto loopOC = scf::ForOp::create(rewriter, loc, c0, boundOC, stepN);
    rewriter.setInsertionPointToStart(loopOC.getBody());
    Value ivOC = loopOC.getInductionVar();

    auto loopOH = scf::ForOp::create(rewriter, loc, c0, boundOH, stepM);
    rewriter.setInsertionPointToStart(loopOH.getBody());
    Value ivOH = loopOH.getInductionVar();

    auto loopOW = scf::ForOp::create(rewriter, loc, c0, boundOW, c1);
    rewriter.setInsertionPointToStart(loopOW.getBody());
    Value ivOW = loopOW.getInductionVar();

    auto loopIC = scf::ForOp::create(rewriter, loc, c0, boundIC, stepK);
    rewriter.setInsertionPointToStart(loopIC.getBody());
    Value ivIC = loopIC.getInductionVar();

    auto loopFH = scf::ForOp::create(rewriter, loc, c0, boundFH, c1);
    rewriter.setInsertionPointToStart(loopFH.getBody());
    Value ivFH = loopFH.getInductionVar();

    auto loopFW = scf::ForOp::create(rewriter, loc, c0, boundFW, c1);
    rewriter.setInsertionPointToStart(loopFW.getBody());
    Value ivFW = loopFW.getInductionVar();

    Value ihBase = arith::MulIOp::create(rewriter, loc, ivOH, strideHVal);
    ihBase = arith::AddIOp::create(rewriter, loc, ihBase, ivFH);
    Value iw = arith::MulIOp::create(rewriter, loc, ivOW, strideWVal);
    iw = arith::AddIOp::create(rewriter, loc, iw, ivFW);

    auto inputTileType = MemRefType::get({TILE_2M, TILE_K}, inputElemType);
    auto filterTileType = MemRefType::get({TILE_K, TILE_N}, inputElemType);
    auto outputTileType =
        MemRefType::get({TILE_M, TILE_N}, rewriter.getI32Type());

    Value inputTile = memref::AllocaOp::create(rewriter, loc, inputTileType);
    Value filterTile = memref::AllocaOp::create(rewriter, loc, filterTileType);
    Value outputTile = memref::AllocaOp::create(rewriter, loc, outputTileType);

    Value tileMBound = arith::ConstantIndexOp::create(rewriter, loc, TILE_M);
    Value tile2MBound = arith::ConstantIndexOp::create(rewriter, loc, TILE_2M);
    Value tileKBound = arith::ConstantIndexOp::create(rewriter, loc, TILE_K);
    Value tileNBound = arith::ConstantIndexOp::create(rewriter, loc, TILE_N);

    auto fillInputLoop = scf::ForOp::create(rewriter, loc, c0, tile2MBound, c1);
    rewriter.setInsertionPointToStart(fillInputLoop.getBody());
    Value fillM = fillInputLoop.getInductionVar();

    auto fillInputInnerLoop =
        scf::ForOp::create(rewriter, loc, c0, tileKBound, c1);
    rewriter.setInsertionPointToStart(fillInputInnerLoop.getBody());
    Value fillK = fillInputInnerLoop.getInductionVar();

    Value mTimesStride =
        arith::MulIOp::create(rewriter, loc, fillM, strideHVal);
    Value inputIH = arith::AddIOp::create(rewriter, loc, ihBase, mTimesStride);
    Value inputIC = arith::AddIOp::create(rewriter, loc, ivIC, fillK);
    Value inputVal = memref::LoadOp::create(
        rewriter, loc, input, ValueRange{ivN, inputIC, inputIH, iw});
    memref::StoreOp::create(rewriter, loc, inputVal, inputTile,
                            ValueRange{fillM, fillK});

    rewriter.setInsertionPointAfter(fillInputLoop);

    auto fillFilterLoop = scf::ForOp::create(rewriter, loc, c0, tileKBound, c1);
    rewriter.setInsertionPointToStart(fillFilterLoop.getBody());
    Value fillFK = fillFilterLoop.getInductionVar();

    auto fillFilterInnerLoop =
        scf::ForOp::create(rewriter, loc, c0, tileNBound, c1);
    rewriter.setInsertionPointToStart(fillFilterInnerLoop.getBody());
    Value fillFN = fillFilterInnerLoop.getInductionVar();

    Value filterOC = arith::AddIOp::create(rewriter, loc, ivOC, fillFN);
    Value filterIC = arith::AddIOp::create(rewriter, loc, ivIC, fillFK);
    Value filterVal = memref::LoadOp::create(
        rewriter, loc, filter, ValueRange{filterOC, filterIC, ivFH, ivFW});
    memref::StoreOp::create(rewriter, loc, filterVal, filterTile,
                            ValueRange{fillFK, fillFN});

    rewriter.setInsertionPointAfter(fillFilterLoop);

    auto loadOutputLoop = scf::ForOp::create(rewriter, loc, c0, tileMBound, c1);
    rewriter.setInsertionPointToStart(loadOutputLoop.getBody());
    Value loadM = loadOutputLoop.getInductionVar();

    auto loadOutputInnerLoop =
        scf::ForOp::create(rewriter, loc, c0, tileNBound, c1);
    rewriter.setInsertionPointToStart(loadOutputInnerLoop.getBody());
    Value loadN = loadOutputInnerLoop.getInductionVar();

    Value outOC = arith::AddIOp::create(rewriter, loc, ivOC, loadN);
    Value outOH = arith::AddIOp::create(rewriter, loc, ivOH, loadM);
    Value outVal = memref::LoadOp::create(rewriter, loc, output,
                                          ValueRange{ivN, outOC, outOH, ivOW});
    memref::StoreOp::create(rewriter, loc, outVal, outputTile,
                            ValueRange{loadM, loadN});

    rewriter.setInsertionPointAfter(loadOutputLoop);

    if (strideH == 1) {
      Vmadot1Op::create(rewriter, loc, outputTile, inputTile, filterTile);
    } else if (strideH == 2) {
      Vmadot2Op::create(rewriter, loc, outputTile, inputTile, filterTile);
    } else if (strideH == 3) {
      Vmadot3Op::create(rewriter, loc, outputTile, inputTile, filterTile);
    } else {
      Value slide = arith::ConstantIntOp::create(rewriter, loc, 0, 64);
      VmadotnOp::create(rewriter, loc, outputTile, inputTile, filterTile,
                        slide);
    }

    auto storeOutputLoop =
        scf::ForOp::create(rewriter, loc, c0, tileMBound, c1);
    rewriter.setInsertionPointToStart(storeOutputLoop.getBody());
    Value storeM = storeOutputLoop.getInductionVar();

    auto storeOutputInnerLoop =
        scf::ForOp::create(rewriter, loc, c0, tileNBound, c1);
    rewriter.setInsertionPointToStart(storeOutputInnerLoop.getBody());
    Value storeN = storeOutputInnerLoop.getInductionVar();

    Value storeOC = arith::AddIOp::create(rewriter, loc, ivOC, storeN);
    Value storeOH = arith::AddIOp::create(rewriter, loc, ivOH, storeM);
    Value storeVal = memref::LoadOp::create(rewriter, loc, outputTile,
                                            ValueRange{storeM, storeN});
    memref::StoreOp::create(rewriter, loc, storeVal, output,
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
  LowerLinalgToIMEPass(const LowerLinalgToIMEPass &other)
      : PassWrapper(other) {}

  Option<std::string> targetCPU{
      *this, "target", llvm::cl::desc("IME target: generic (K1) or k3 (A100)"),
      llvm::cl::init("generic")};
  Option<std::string> matmulShapes{
      *this, "matmul-shapes",
      llvm::cl::desc("Optional comma-separated KxN matmul shapes to lower"),
      llvm::cl::init("")};
  Option<int64_t> packedVectorSize{
      *this, "packed-vector-size",
      llvm::cl::desc("Consume panel-packed decode weights (0 means row-major)"),
      llvm::cl::init(0)};
  Option<bool> parallelTiles{
      *this, "parallel-tiles",
      llvm::cl::desc("Parallelize independent N tiles with private scratch"),
      llvm::cl::init(false)};
  Option<bool> verifyFP16{
      *this, "verify-fp16",
      llvm::cl::desc("Check a nonzero FP16 IME product before model execution"),
      llvm::cl::init(false)};
  Option<bool> a100FP16{
      *this, "a100-fp16",
      llvm::cl::desc(
          "Use A100 8x8x8 FP16 input / FP32 accumulation instructions"),
      llvm::cl::init(false)};

  Option<bool> a100Gemv{
      *this, "a100-gemv",
      llvm::cl::desc("Use register-resident GEMV and offline N32/K8 tiles"),
      llvm::cl::init(false)};

  Option<bool> a100Gemm{
      *this, "a100-gemm",
      llvm::cl::desc(
          "Use packed Prefill panels and register-resident 8x32 GEMM"),
      llvm::cl::init(false)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IMEDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<cf::ControlFlowDialect>();
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LowerLinalgToIMEPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  RewritePatternSet patterns(context);

  if (targetCPU != "generic" && targetCPU != "k3") {
    module.emitError("unsupported IME target: ") << targetCPU;
    return signalPassFailure();
  }
  bool k3 = targetCPU == "k3";
  if (packedVectorSize < 0) {
    module.emitError("packed-vector-size must be nonnegative");
    return signalPassFailure();
  }
  if ((a100FP16 && !k3) || (a100Gemv && a100Gemm) ||
      ((a100Gemm || a100Gemv) && !a100FP16) || (a100Gemm && packedVectorSize)) {
    module.emitError(
        "A100 FP16 kernels require target=k3 and compatible kernel options");
    return signalPassFailure();
  }
  patterns.add<MatmulToIMELowering>(context, k3, matmulShapes, packedVectorSize,
                                    parallelTiles, verifyFP16, a100FP16,
                                    a100Gemv, a100Gemm);
  // Selected packed matmuls must not redirect attention or generic operations.
  if (matmulShapes.empty() && packedVectorSize == 0 && !a100FP16) {
    patterns
        .add<GenericMatmulToIMELowering, BatchMatmulTransposeBToIMELowering>(
            context, k3);
  }
  if (!k3) {
    patterns.add<Conv2DNhwcHwcfToIMELowering>(context);
    patterns.add<Conv2DNchwFchwToIMELowering>(context);
  }

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
