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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialect/IME/IMEDialect.h"
#include "Dialect/IME/IMEOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

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
static void insertFP16IMECheck(RewriterBase &rewriter, Location loc,
                               ModuleOp module, bool a100,
                               func::FuncOp caller) {
  // C-wrapper export makes helper symbols external. Scope each helper to its
  // graph so separately compiled Prefill and Decode partitions can be linked.
  std::string name =
      (a100 ? "__buddy_check_a100_fp16_ime_" : "__buddy_check_fp16_ime_") +
      caller.getSymName().str();
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
    rewriter.setInsertionPointToStart(&caller.front());
    func::CallOp::create(rewriter, loc, name, TypeRange{}, ValueRange{});
  }
}

// Establish dynamic strides once, before the output-tile loop. Static
// incompatibilities are diagnosed by the GEMV op verifier. Preserve offsets.
static Value checkGemvStrides(PatternRewriter &rewriter, Location loc,
                              Value value) {
  auto type = cast<MemRefType>(value.getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return value;
  ValueRange actual;
  for (int dim = 0; dim < 2; ++dim) {
    if (type.getDimSize(dim) == 1 || !ShapedType::isDynamic(strides[dim]))
      continue;
    if (actual.empty())
      actual = memref::ExtractStridedMetadataOp::create(rewriter, loc, value)
                   .getStrides();
    strides[dim] = dim == 0 ? type.getDimSize(1) : 1;
    Value expected =
        arith::ConstantIndexOp::create(rewriter, loc, strides[dim]);
    Value matches = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, actual[dim], expected);
    cf::AssertOp::create(rewriter, loc, matches,
                         "A100 GEMV requires contiguous operands");
  }
  if (actual.empty())
    return value;
  auto layout = StridedLayoutAttr::get(rewriter.getContext(), offset, strides);
  auto checked = MemRefType::get(type.getShape(), type.getElementType(), layout,
                                 type.getMemorySpace());
  return memref::CastOp::create(rewriter, loc, checked, value);
}

struct TransposedBInfo {
  Value source;
  linalg::TransposeOp transposeOp;
  memref::AllocOp allocOp;
  SmallVector<memref::DeallocOp> deallocOps;
};

static bool hasUnitInnerStride(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset;
  return type.hasStaticShape() && type.getRank() >= 1 &&
         succeeded(type.getStridesAndOffset(strides, offset)) &&
         strides.back() == 1;
}

static std::optional<TransposedBInfo> tryFindTransposedB(Value b,
                                                         Operation *matmulOp) {
  auto allocOp = b.getDefiningOp<memref::AllocOp>();
  if (!allocOp)
    return std::nullopt;
  auto bType = dyn_cast<MemRefType>(b.getType());
  if (!bType || !bType.hasStaticShape() || bType.getRank() != 2)
    return std::nullopt;

  TransposedBInfo info;
  info.allocOp = allocOp;
  bool seenMatmul = false;
  for (Operation *user : b.getUsers()) {
    if (user == matmulOp) {
      if (seenMatmul)
        return std::nullopt;
      seenMatmul = true;
      continue;
    }
    if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      info.deallocOps.push_back(dealloc);
      continue;
    }
    auto transposeOp = dyn_cast<linalg::TransposeOp>(user);
    if (!transposeOp || transposeOp->getBlock() != matmulOp->getBlock() ||
        transposeOp.getInput() == Value() || transposeOp.getInit() != b ||
        info.transposeOp)
      return std::nullopt;
    ArrayRef<int64_t> perm = transposeOp.getPermutation();
    if (perm != ArrayRef<int64_t>{1, 0})
      return std::nullopt;
    info.transposeOp = transposeOp;
    info.source = transposeOp.getInput();
  }
  if (!seenMatmul || !info.transposeOp)
    return std::nullopt;
  auto srcType = dyn_cast<MemRefType>(info.source.getType());
  if (!srcType || !srcType.hasStaticShape() || srcType.getRank() != 2 ||
      srcType.getElementType() != bType.getElementType() ||
      srcType.getDimSize(0) != bType.getDimSize(1) ||
      srcType.getDimSize(1) != bType.getDimSize(0))
    return std::nullopt;
  return info;
}

static void eraseDeadDeallocAndAlloc(RewriterBase &rewriter,
                                     memref::AllocOp allocOp,
                                     ArrayRef<memref::DeallocOp> deallocOps) {
  for (memref::DeallocOp dealloc : deallocOps) {
    Operation *parent = dealloc->getParentOp();
    rewriter.eraseOp(dealloc);
    if (auto ifOp = dyn_cast_or_null<scf::IfOp>(parent)) {
      if (ifOp.getNumResults() == 0 && ifOp.getElseRegion().empty() &&
          ifOp.thenBlock()->without_terminator().empty())
        rewriter.eraseOp(ifOp);
    }
  }
  if (allocOp && allocOp->use_empty())
    rewriter.eraseOp(allocOp);
}

static Value createZeroCopyTransposedView(OpBuilder &builder, Location loc,
                                          Value src) {
  AffineExpr d0 = builder.getAffineDimExpr(0);
  AffineExpr d1 = builder.getAffineDimExpr(1);
  AffineMap perm = AffineMap::get(2, 0, {d1, d0}, builder.getContext());
  return memref::TransposeOp::create(builder, loc, src,
                                     AffineMapAttr::get(perm));
}

static void emitZeroFill1DF16(OpBuilder &builder, Location loc, Value buf,
                              int64_t elements) {
  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);
  Value zeroF16 = arith::ConstantOp::create(
      builder, loc, builder.getF16Type(),
      builder.getFloatAttr(builder.getF16Type(), 0.0));
  int64_t vecEnd = (elements / 32) * 32;
  if (vecEnd > 0) {
    Value vEnd = arith::ConstantIndexOp::create(builder, loc, vecEnd);
    Value c32 = arith::ConstantIndexOp::create(builder, loc, 32);
    auto vecTy = VectorType::get({32}, builder.getF16Type());
    Value vZero = vector::BroadcastOp::create(builder, loc, vecTy, zeroF16);
    auto loop = scf::ForOp::create(builder, loc, c0, vEnd, c32);
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(loop.getBody());
    vector::StoreOp::create(builder, loc, vZero, buf,
                            ValueRange{loop.getInductionVar()});
  }
  if (vecEnd < elements) {
    Value vEnd = arith::ConstantIndexOp::create(builder, loc, vecEnd);
    Value total = arith::ConstantIndexOp::create(builder, loc, elements);
    auto tail = scf::ForOp::create(builder, loc, vEnd, total, c1);
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(tail.getBody());
    memref::StoreOp::create(builder, loc, zeroF16, buf,
                            ValueRange{tail.getInductionVar()});
  }
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
    std::optional<TransposedBInfo> transposedB;
    bool isM1TransposedGemv = false;
    if (fastGemm) {
      transposedB = tryFindTransposedB(B, matmulOp);
      if (M == 1) {
        if (!a100 || !isFloatMatmul || !CElemType.isF16() || packedWidth ||
            K % 64 != 0 || !hasUnitInnerStride(AType) ||
            !CType.getLayout().isIdentity() || !transposedB)
          return rewriter.notifyMatchFailure(
              matmulOp,
              "A100 GEMM requires M>=8 or M=1 (K%64=0) with transposed [N,K] "
              "weights");
        isM1TransposedGemv = true;
      } else if (!a100 || !isFloatMatmul || !CElemType.isF16() || packedWidth ||
                 M < 8 || K % 8 != 0 || N < 32 ||
                 !AType.getLayout().isIdentity() ||
                 !CType.getLayout().isIdentity()) {
        return rewriter.notifyMatchFailure(
            matmulOp,
            "A100 GEMM requires contiguous FP16 A/C, M>=8, K%8=0, N>=32");
      }
    }
    if (verifyFP16 && isFloatMatmul)
      insertFP16IMECheck(rewriter, loc, matmulOp->getParentOfType<ModuleOp>(),
                         a100, matmulOp->getParentOfType<func::FuncOp>());

    if (fastGemm && isM1TransposedGemv) {
      Value src = transposedB->source;
      auto meta = memref::ExtractStridedMetadataOp::create(rewriter, loc, src);
      Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value stride1 = meta.getStrides()[1];
      Value isUnit = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, stride1, one);
      cf::AssertOp::create(
          rewriter, loc, isUnit,
          "M=1 transposed [N,K] GEMV requires unit inner weight stride");
      auto flatLayout =
          StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic,
                                 ArrayRef<int64_t>{ShapedType::kDynamic, 1});
      auto flatType = MemRefType::get({N, K}, BElemType, flatLayout);
      Value flatSrc = memref::ReinterpretCastOp::create(
          rewriter, loc, flatType, meta.getBaseBuffer(), meta.getOffset(),
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(N),
                                    rewriter.getIndexAttr(K)},
          SmallVector<OpFoldResult>{meta.getStrides()[0],
                                    rewriter.getIndexAttr(1)});

      Value boundN = arith::ConstantIndexOp::create(rewriter, loc, N);
      Value boundK = arith::ConstantIndexOp::create(rewriter, loc, K);
      Value c32 = arith::ConstantIndexOp::create(rewriter, loc, 32);
      Value c64 = arith::ConstantIndexOp::create(rewriter, loc, 64);
      auto vec16Ty = VectorType::get({32}, AElemType);
      auto vec32Ty = VectorType::get({32}, rewriter.getF32Type());
      Value zeroF32 = arith::ConstantOp::create(
          rewriter, loc, rewriter.getF32Type(),
          rewriter.getFloatAttr(rewriter.getF32Type(), 0.0));
      Value zeroVec =
          vector::BroadcastOp::create(rewriter, loc, vec32Ty, zeroF32);

      auto nLoop = scf::ParallelOp::create(rewriter, loc, ValueRange{zero},
                                           ValueRange{boundN}, ValueRange{one});
      rewriter.setInsertionPointToStart(nLoop.getBody());
      Value nIdx = nLoop.getInductionVars()[0];
      auto kLoop = scf::ForOp::create(rewriter, loc, zero, boundK, c64,
                                      ValueRange{zeroVec, zeroVec});
      rewriter.setInsertionPointToStart(kLoop.getBody());
      Value kIdx = kLoop.getInductionVar();
      Value kHalf = arith::AddIOp::create(rewriter, loc, kIdx, c32);
      Value aLo = vector::LoadOp::create(rewriter, loc, vec16Ty, A,
                                         ValueRange{zero, kIdx});
      Value aHi = vector::LoadOp::create(rewriter, loc, vec16Ty, A,
                                         ValueRange{zero, kHalf});
      Value bLo = vector::LoadOp::create(rewriter, loc, vec16Ty, flatSrc,
                                         ValueRange{nIdx, kIdx});
      Value bHi = vector::LoadOp::create(rewriter, loc, vec16Ty, flatSrc,
                                         ValueRange{nIdx, kHalf});
      Value aLo32 = arith::ExtFOp::create(rewriter, loc, vec32Ty, aLo);
      Value aHi32 = arith::ExtFOp::create(rewriter, loc, vec32Ty, aHi);
      Value bLo32 = arith::ExtFOp::create(rewriter, loc, vec32Ty, bLo);
      Value bHi32 = arith::ExtFOp::create(rewriter, loc, vec32Ty, bHi);
      Value accLo = vector::FMAOp::create(rewriter, loc, aLo32, bLo32,
                                          kLoop.getRegionIterArgs()[0]);
      Value accHi = vector::FMAOp::create(rewriter, loc, aHi32, bHi32,
                                          kLoop.getRegionIterArgs()[1]);
      scf::YieldOp::create(rewriter, loc, ValueRange{accLo, accHi});

      rewriter.setInsertionPointAfter(kLoop);
      Value sumVec = arith::AddFOp::create(rewriter, loc, kLoop.getResult(0),
                                           kLoop.getResult(1));
      Value dot = vector::ReductionOp::create(
          rewriter, loc, vector::CombiningKind::ADD, sumVec);
      Value cVal =
          memref::LoadOp::create(rewriter, loc, C, ValueRange{zero, nIdx});
      Value cF32 =
          arith::ExtFOp::create(rewriter, loc, rewriter.getF32Type(), cVal);
      Value total = arith::AddFOp::create(rewriter, loc, cF32, dot);
      Value outVal = arith::TruncFOp::create(rewriter, loc, CElemType, total);
      memref::StoreOp::create(rewriter, loc, outVal, C, ValueRange{zero, nIdx});

      rewriter.setInsertionPointAfter(nLoop);
      TransposedBInfo info = *transposedB;
      rewriter.eraseOp(matmulOp);
      rewriter.eraseOp(info.transposeOp);
      eraseDeadDeallocAndAlloc(rewriter, info.allocOp, info.deallocOps);
      return success();
    }

    if (fastGemm) {
      Value bSource = B;
      if (transposedB)
        bSource =
            createZeroCopyTransposedView(rewriter, loc, transposedB->source);
      auto index = [&](int64_t value) -> Value {
        return arith::ConstantIndexOp::create(rewriter, loc, value);
      };
      const int64_t tiledM = (M / 8) * 8;
      const int64_t tiledN = (N / 32) * 32;
      Value zero = index(0), eight = index(8), one = index(1);
      Value boundM = index(tiledM), boundK = index(K), boundN = index(tiledN);
      Value packedA = memref::AllocOp::create(
          rewriter, loc, MemRefType::get({tiledM * K}, AElemType));
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
      PackBF16Op::create(rewriter, loc, packedB, bSource, column,
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
      // Compute M % 8 or N % 32 tail elements with the same FP32 accumulation
      // used by GemmF16Op.
      auto emitFP32Tail = [&](int64_t rowBegin, int64_t rowEnd,
                              int64_t colBegin, int64_t colEnd) {
        auto tail = scf::ParallelOp::create(
            rewriter, loc, ValueRange{index(rowBegin), index(colBegin)},
            ValueRange{index(rowEnd), index(colEnd)}, ValueRange{one, one});
        rewriter.setInsertionPointToStart(tail.getBody());
        Value i = tail.getInductionVars()[0];
        Value j = tail.getInductionVars()[1];
        Value current =
            memref::LoadOp::create(rewriter, loc, C, ValueRange{i, j});
        Value initial = arith::ExtFOp::create(rewriter, loc,
                                              rewriter.getF32Type(), current);
        auto reduction = scf::ForOp::create(rewriter, loc, zero, boundK, one,
                                            ValueRange{initial});
        rewriter.setInsertionPointToStart(reduction.getBody());
        Value k = reduction.getInductionVar();
        Value lhs = memref::LoadOp::create(rewriter, loc, A, ValueRange{i, k});
        Value rhs =
            memref::LoadOp::create(rewriter, loc, bSource, ValueRange{k, j});
        Value lhsF32 =
            arith::ExtFOp::create(rewriter, loc, rewriter.getF32Type(), lhs);
        Value rhsF32 =
            arith::ExtFOp::create(rewriter, loc, rewriter.getF32Type(), rhs);
        Value product = arith::MulFOp::create(rewriter, loc, lhsF32, rhsF32);
        Value sum = arith::AddFOp::create(
            rewriter, loc, reduction.getRegionIterArgs()[0], product);
        scf::YieldOp::create(rewriter, loc, sum);
        rewriter.setInsertionPointAfter(reduction);
        Value result = arith::TruncFOp::create(
            rewriter, loc, rewriter.getF16Type(), reduction.getResult(0));
        memref::StoreOp::create(rewriter, loc, result, C, ValueRange{i, j});
        rewriter.setInsertionPointAfter(tail);
      };
      if (tiledN != N)
        emitFP32Tail(0, tiledM, tiledN, N);
      if (tiledM != M)
        emitFP32Tail(tiledM, M, 0, N);

      rewriter.eraseOp(matmulOp);
      if (transposedB) {
        rewriter.eraseOp(transposedB->transposeOp);
        eraseDeadDeallocAndAlloc(rewriter, transposedB->allocOp,
                                 transposedB->deallocOps);
      }
      return success();
    }

    if (fastGemv) {
      if (!a100 || !isFloatMatmul || !CElemType.isF16() || M != 1 ||
          packedWidth < 8 || packedWidth > 32 || packedWidth % 8 || K % 8 ||
          N % packedWidth)
        return matmulOp.emitError("A100 fast GEMV requires FP16 M=1, K%8=0, "
                                  "and aligned 8/16/24/32-column panels");
      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value bound = arith::ConstantIndexOp::create(rewriter, loc, N);
      Value step = arith::ConstantIndexOp::create(rewriter, loc, packedWidth);
      A = checkGemvStrides(rewriter, loc, A);
      B = checkGemvStrides(rewriter, loc, B);
      C = checkGemvStrides(rewriter, loc, C);
      bool unroll = K % 16 == 0 && (N <= 256 || K == N);
      int64_t prefetch = N <= 256 ? 0 : (unroll ? 1024 : 512);
      auto loop = scf::ParallelOp::create(rewriter, loc, ValueRange{zero},
                                          ValueRange{bound}, ValueRange{step});
      rewriter.setInsertionPointToStart(loop.getBody());
      GemvF16Op::create(rewriter, loc, C, A, B, loop.getInductionVars()[0],
                        rewriter.getI64IntegerAttr(K),
                        rewriter.getI64IntegerAttr(prefetch),
                        rewriter.getBoolAttr(unroll),
                        rewriter.getI64IntegerAttr(packedWidth));
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
      llvm::cl::desc("Use register-resident GEMV with offline K8 panels"),
      llvm::cl::init(false)};

  Option<bool> a100Gemm{
      *this, "a100-gemm",
      llvm::cl::desc(
          "Use packed Prefill panels and register-resident 8x32 GEMM"),
      llvm::cl::init(false)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IMEDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<cf::ControlFlowDialect>();
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

static Value stripToZeroInitOp(Value buf, Operation *consumerOp,
                               Operation *&initOpOut) {
  initOpOut = nullptr;
  auto alloc = buf.getDefiningOp<memref::AllocOp>();
  if (!alloc)
    return Value{};
  for (Operation *user : buf.getUsers()) {
    if (user == consumerOp || isa<memref::DeallocOp>(user))
      continue;
    if (user->getBlock() != consumerOp->getBlock())
      return Value{};
    if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
      if (copyOp.getTarget() != buf || initOpOut ||
          !copyOp->isBeforeInBlock(consumerOp))
        return Value{};
      initOpOut = copyOp;
      continue;
    }
    if (auto fillOp = dyn_cast<linalg::FillOp>(user)) {
      if (fillOp.getOutputs().size() != 1 || fillOp.getOutputs()[0] != buf ||
          initOpOut || !fillOp->isBeforeInBlock(consumerOp))
        return Value{};
      initOpOut = fillOp;
      continue;
    }
  }
  return buf;
}

static bool isDefaultBmmTransposeBMaps(linalg::BatchMatmulOp op) {
  auto maps = op.getIndexingMapsArray();
  if (maps.size() != 3)
    return false;
  MLIRContext *ctx = op.getContext();
  AffineExpr b = getAffineDimExpr(0, ctx), m = getAffineDimExpr(1, ctx);
  AffineExpr n = getAffineDimExpr(2, ctx), d = getAffineDimExpr(3, ctx);
  return maps[0] == AffineMap::get(4, 0, {b, m, d}, ctx) &&
         maps[1] == AffineMap::get(4, 0, {b, n, d}, ctx) &&
         maps[2] == AffineMap::get(4, 0, {b, m, n}, ctx);
}

static bool isDefaultBmmMaps(linalg::BatchMatmulOp op) {
  auto maps = op.getIndexingMapsArray();
  if (maps.size() != 3)
    return false;
  MLIRContext *ctx = op.getContext();
  AffineExpr b = getAffineDimExpr(0, ctx), m = getAffineDimExpr(1, ctx);
  AffineExpr n = getAffineDimExpr(2, ctx), k = getAffineDimExpr(3, ctx);
  return maps[0] == AffineMap::get(4, 0, {b, m, k}, ctx) &&
         maps[1] == AffineMap::get(4, 0, {b, k, n}, ctx) &&
         maps[2] == AffineMap::get(4, 0, {b, m, n}, ctx);
}

static bool isIdentity3DMap(AffineMap map) {
  if (map.getNumDims() != 3 || map.getNumResults() != 3)
    return false;
  return map == AffineMap::getMultiDimIdentityMap(3, map.getContext());
}

static bool isMask3DMap(AffineMap map) {
  if (map.getNumDims() != 3 || map.getNumResults() != 3)
    return false;
  MLIRContext *ctx = map.getContext();
  return map ==
         AffineMap::get(3, 0,
                        {getAffineConstantExpr(0, ctx),
                         getAffineDimExpr(1, ctx), getAffineDimExpr(2, ctx)},
                        ctx);
}

static void rewritePatchEmbeddingConvs(ModuleOp module, bool verifyFP16) {
  SmallVector<linalg::Conv2DNhwcFhwcOp> convs;
  module.walk([&](linalg::Conv2DNhwcFhwcOp conv) { convs.push_back(conv); });
  IRRewriter rewriter(module.getContext());

  for (linalg::Conv2DNhwcFhwcOp conv : convs) {
    if (conv.getNumDpsInputs() != 2 || conv.getNumDpsInits() != 1)
      continue;
    Value input = conv.getInputs()[0];
    Value filter = conv.getInputs()[1];
    Value output = conv.getOutputs()[0];
    auto inTy = dyn_cast<MemRefType>(input.getType());
    auto filTy = dyn_cast<MemRefType>(filter.getType());
    auto outTy = dyn_cast<MemRefType>(output.getType());
    if (!inTy || !filTy || !outTy || !inTy.hasStaticShape() ||
        !filTy.hasStaticShape() || !outTy.hasStaticShape() ||
        inTy.getRank() != 4 || filTy.getRank() != 4 || outTy.getRank() != 4 ||
        !inTy.getElementType().isF16() || !filTy.getElementType().isF16() ||
        !outTy.getElementType().isF16() || !inTy.getLayout().isIdentity() ||
        !filTy.getLayout().isIdentity() || !outTy.getLayout().isIdentity())
      continue;

    int64_t M = inTy.getDimSize(0), H = inTy.getDimSize(1),
            W = inTy.getDimSize(2), C = inTy.getDimSize(3);
    int64_t N = filTy.getDimSize(0), KH = filTy.getDimSize(1),
            KW = filTy.getDimSize(2), FC = filTy.getDimSize(3);
    if (H != KH || W != KW || C != FC || outTy.getDimSize(0) != M ||
        outTy.getDimSize(1) != 1 || outTy.getDimSize(2) != 1 ||
        outTy.getDimSize(3) != N || M < 8 || M % 8 != 0 || N < 32 ||
        N % 32 != 0)
      continue;

    auto strides = conv.getStrides();
    auto dilations = conv.getDilations();
    if (!strides || !dilations)
      continue;
    SmallVector<int64_t> strideVals(strides.getValues<int64_t>());
    SmallVector<int64_t> dilVals(dilations.getValues<int64_t>());
    if (strideVals != SmallVector<int64_t>{H, W} ||
        dilVals != SmallVector<int64_t>{1, 1})
      continue;

    Location loc = conv.getLoc();
    rewriter.setInsertionPoint(conv);
    if (verifyFP16)
      insertFP16IMECheck(rewriter, loc, module, /*a100=*/true,
                         conv->getParentOfType<func::FuncOp>());

    int64_t K = H * W * C;
    int64_t completeK = (K / 8) * 8;
    int64_t paddedK = ((K + 7) / 8) * 8;
    Type f16 = rewriter.getF16Type();
    auto index = [&](int64_t v) -> Value {
      return arith::ConstantIndexOp::create(rewriter, loc, v);
    };
    Value c0 = index(0), c1 = index(1), c8 = index(8), c32 = index(32);
    Value boundM = index(M), boundN = index(N), boundPaddedK = index(paddedK);
    Value zeroF16 = arith::ConstantOp::create(rewriter, loc, f16,
                                              rewriter.getFloatAttr(f16, 0.0));

    SmallVector<ReassociationIndices> reassoc = {{0}, {1, 2, 3}};
    Value A2D = memref::CollapseShapeOp::create(rewriter, loc, input, reassoc);
    Value F2D = memref::CollapseShapeOp::create(rewriter, loc, filter, reassoc);
    Value C2D = memref::CollapseShapeOp::create(rewriter, loc, output, reassoc);

    Value packedA = memref::AllocOp::create(
        rewriter, loc, MemRefType::get({M * paddedK}, f16));
    auto packRows = scf::ParallelOp::create(rewriter, loc, ValueRange{c0},
                                            ValueRange{boundM}, ValueRange{c8});
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(packRows.getBody());
      Value row = packRows.getInductionVars()[0];
      Value rowBase = arith::MulIOp::create(rewriter, loc, row, boundPaddedK);
      if (completeK > 0) {
        Value vCompleteK = index(completeK);
        auto packK = scf::ForOp::create(rewriter, loc, c0, vCompleteK, c8);
        rewriter.setInsertionPointToStart(packK.getBody());
        Value kb = packK.getInductionVar();
        Value blk = arith::AddIOp::create(
            rewriter, loc, arith::MulIOp::create(rewriter, loc, kb, c8),
            rowBase);
        auto packRow = scf::ForOp::create(rewriter, loc, c0, c8, c1);
        rewriter.setInsertionPointToStart(packRow.getBody());
        Value r = packRow.getInductionVar();
        Value srcRow = arith::AddIOp::create(rewriter, loc, row, r);
        Value dstOff = arith::AddIOp::create(
            rewriter, loc, arith::MulIOp::create(rewriter, loc, r, c8), blk);
        Value vec =
            vector::LoadOp::create(rewriter, loc, VectorType::get({8}, f16),
                                   A2D, ValueRange{srcRow, kb});
        vector::StoreOp::create(rewriter, loc, vec, packedA,
                                ValueRange{dstOff});
      }
      if (completeK < K) {
        rewriter.setInsertionPoint(packRows.getBody()->getTerminator());
        Value blk = arith::AddIOp::create(
            rewriter, loc,
            arith::MulIOp::create(rewriter, loc, index(completeK), c8),
            rowBase);
        Value remK = index(K - completeK);
        Value vCompleteK = index(completeK);
        auto tailR = scf::ForOp::create(rewriter, loc, c0, c8, c1);
        rewriter.setInsertionPointToStart(tailR.getBody());
        Value r = tailR.getInductionVar();
        Value srcRow = arith::AddIOp::create(rewriter, loc, row, r);
        Value rowOff = arith::AddIOp::create(
            rewriter, loc, arith::MulIOp::create(rewriter, loc, r, c8), blk);
        auto tailK = scf::ForOp::create(rewriter, loc, c0, c8, c1);
        rewriter.setInsertionPointToStart(tailK.getBody());
        Value ko = tailK.getInductionVar();
        Value inBound = arith::CmpIOp::create(
            rewriter, loc, arith::CmpIPredicate::ult, ko, remK);
        auto ifOp = scf::IfOp::create(rewriter, loc, f16, inBound, true);
        rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
        Value srcK = arith::AddIOp::create(rewriter, loc, vCompleteK, ko);
        Value v = memref::LoadOp::create(rewriter, loc, A2D,
                                         ValueRange{srcRow, srcK});
        scf::YieldOp::create(rewriter, loc, v);
        rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
        scf::YieldOp::create(rewriter, loc, zeroF16);
        rewriter.setInsertionPointAfter(ifOp);
        Value dstOff = arith::AddIOp::create(rewriter, loc, rowOff, ko);
        memref::StoreOp::create(rewriter, loc, ifOp.getResult(0), packedA,
                                ValueRange{dstOff});
      }
    }

    auto columns = scf::ParallelOp::create(rewriter, loc, ValueRange{c0},
                                           ValueRange{boundN}, ValueRange{c32});
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(columns.getBody());
      Value col = columns.getInductionVars()[0];
      Value packedB = memref::AllocOp::create(
          rewriter, loc, MemRefType::get({paddedK * 32}, f16));
      if (completeK > 0) {
        OpFoldResult idx0 = rewriter.getIndexAttr(0);
        OpFoldResult idx1 = rewriter.getIndexAttr(1);
        Value fSub = memref::SubViewOp::create(
            rewriter, loc, F2D, SmallVector<OpFoldResult>{idx0, idx0},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(N),
                                      rewriter.getIndexAttr(completeK)},
            SmallVector<OpFoldResult>{idx1, idx1});
        Value fTrans = createZeroCopyTransposedView(rewriter, loc, fSub);
        Value pbSub = memref::SubViewOp::create(
            rewriter, loc, packedB, SmallVector<OpFoldResult>{idx0},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(completeK * 32)},
            SmallVector<OpFoldResult>{idx1});
        PackBF16Op::create(rewriter, loc, pbSub, fTrans, col,
                           rewriter.getI64IntegerAttr(completeK));
      }
      if (completeK < K) {
        Value tailOff = index(completeK * 32);
        Value remK = index(K - completeK);
        Value vCompleteK = index(completeK);
        auto tailC = scf::ForOp::create(rewriter, loc, c0, c32, c1);
        rewriter.setInsertionPointToStart(tailC.getBody());
        Value c = tailC.getInductionVar();
        Value srcCol = arith::AddIOp::create(rewriter, loc, col, c);
        Value colOff =
            arith::AddIOp::create(rewriter, loc, tailOff,
                                  arith::MulIOp::create(rewriter, loc, c, c8));
        auto tailK = scf::ForOp::create(rewriter, loc, c0, c8, c1);
        rewriter.setInsertionPointToStart(tailK.getBody());
        Value ko = tailK.getInductionVar();
        Value inBound = arith::CmpIOp::create(
            rewriter, loc, arith::CmpIPredicate::ult, ko, remK);
        auto ifOp = scf::IfOp::create(rewriter, loc, f16, inBound, true);
        rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
        Value srcK = arith::AddIOp::create(rewriter, loc, vCompleteK, ko);
        Value v = memref::LoadOp::create(rewriter, loc, F2D,
                                         ValueRange{srcCol, srcK});
        scf::YieldOp::create(rewriter, loc, v);
        rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
        scf::YieldOp::create(rewriter, loc, zeroF16);
        rewriter.setInsertionPointAfter(ifOp);
        Value dstOff = arith::AddIOp::create(rewriter, loc, colOff, ko);
        memref::StoreOp::create(rewriter, loc, ifOp.getResult(0), packedB,
                                ValueRange{dstOff});
      }
      rewriter.setInsertionPoint(columns.getBody()->getTerminator());
      auto rows = scf::ForOp::create(rewriter, loc, c0, boundM, c8);
      rewriter.setInsertionPointToStart(rows.getBody());
      GemmF16Op::create(
          rewriter, loc, C2D, packedA, packedB, rows.getInductionVar(), col,
          rewriter.getI64IntegerAttr(paddedK), rewriter.getI64IntegerAttr(N));
      rewriter.setInsertionPointAfter(rows);
      memref::DeallocOp::create(rewriter, loc, packedB);
    }
    memref::DeallocOp::create(rewriter, loc, packedA);
    rewriter.eraseOp(conv);
  }
}

namespace {
struct FlashAttentionChain {
  linalg::BatchMatmulOp bmm1;
  Operation *bmm1Init = nullptr;
  linalg::GenericOp scaleOp;
  linalg::FillOp maxFill;
  linalg::ReduceOp maxReduce;
  Operation *maxView = nullptr;
  linalg::GenericOp exp1Op;
  linalg::FillOp sumFill;
  linalg::ReduceOp sumReduce;
  Operation *sumView = nullptr;
  linalg::GenericOp logAddOp;
  linalg::GenericOp exp2Op;
  linalg::FillOp bmm2Fill;
  linalg::BatchMatmulOp bmm2;

  Value Q, K, V, mask, O;
  int64_t B = 0, M = 0, N = 0, D = 0;

  SmallVector<Value> deadBuffers;
  SmallVector<memref::AllocOp> deadAllocs;
  SmallVector<memref::DeallocOp> deadDeallocs;
};
} // namespace

static bool isPureViewOf(Operation *op, Value src) {
  if (auto r = dyn_cast<memref::ReinterpretCastOp>(op))
    return r.getSource() == src;
  if (auto e = dyn_cast<memref::ExpandShapeOp>(op))
    return e.getSrc() == src;
  return false;
}

static bool matchFlashAttentionChain(linalg::BatchMatmulOp bmm1,
                                     FlashAttentionChain &chain) {
  if (!isDefaultBmmTransposeBMaps(bmm1) || bmm1.getNumDpsInputs() != 2 ||
      bmm1.getNumDpsInits() != 1)
    return false;
  Value Q = bmm1.getInputs()[0], K = bmm1.getInputs()[1],
        S0 = bmm1.getOutputs()[0];
  auto qTy = dyn_cast<MemRefType>(Q.getType()),
       kTy = dyn_cast<MemRefType>(K.getType()),
       s0Ty = dyn_cast<MemRefType>(S0.getType());
  if (!qTy || !kTy || !s0Ty || !qTy.hasStaticShape() || !kTy.hasStaticShape() ||
      !s0Ty.hasStaticShape() || qTy.getRank() != 3 || kTy.getRank() != 3 ||
      s0Ty.getRank() != 3 || !qTy.getElementType().isF16() ||
      !kTy.getElementType().isF16() || !s0Ty.getElementType().isF16() ||
      !qTy.getLayout().isIdentity() || !kTy.getLayout().isIdentity() ||
      !s0Ty.getLayout().isIdentity())
    return false;

  int64_t B = qTy.getDimSize(0), M = qTy.getDimSize(1), D = qTy.getDimSize(2);
  int64_t N = kTy.getDimSize(1);
  if (kTy.getDimSize(0) != B || kTy.getDimSize(2) != D ||
      s0Ty.getDimSize(0) != B || s0Ty.getDimSize(1) != M ||
      s0Ty.getDimSize(2) != N || D < 8 || D % 8 != 0 || M < 1 || N < 1)
    return false;

  Operation *bmm1Init = nullptr;
  if (!stripToZeroInitOp(S0, bmm1, bmm1Init) || !bmm1Init)
    return false;

  // Find scaleOp consuming S0.
  linalg::GenericOp scaleOp;
  for (Operation *u : S0.getUsers()) {
    if (u == bmm1 || u == bmm1Init || isa<memref::DeallocOp>(u))
      continue;
    if (auto g = dyn_cast<linalg::GenericOp>(u)) {
      if (scaleOp)
        return false;
      scaleOp = g;
    } else {
      return false;
    }
  }
  if (!scaleOp || scaleOp->getBlock() != bmm1->getBlock() ||
      scaleOp.getNumDpsInits() != 1 || scaleOp.getNumLoops() != 3 ||
      scaleOp.getNumParallelLoops() != 3)
    return false;
  auto scaleMaps = scaleOp.getIndexingMapsArray();
  if (scaleOp.getNumDpsInputs() < 1 || scaleOp.getNumDpsInputs() > 2 ||
      scaleOp.getInputs()[0] != S0 || !isIdentity3DMap(scaleMaps[0]) ||
      !isIdentity3DMap(scaleMaps.back()))
    return false;
  Value mask;
  if (scaleOp.getNumDpsInputs() == 2) {
    mask = scaleOp.getInputs()[1];
    auto maskTy = dyn_cast<MemRefType>(mask.getType());
    if (!maskTy || !maskTy.hasStaticShape() || maskTy.getRank() != 3 ||
        !maskTy.getElementType().isF16() || maskTy.getDimSize(0) != 1 ||
        maskTy.getDimSize(1) != M || maskTy.getDimSize(2) != N ||
        !isMask3DMap(scaleMaps[1]))
      return false;
  }
  Value S1 = scaleOp.getOutputs()[0];
  auto s1Alloc = S1.getDefiningOp<memref::AllocOp>();
  if (!s1Alloc || S1.getType() != s0Ty)
    return false;

  // From S1: locate maxReduce, exp1Op, exp2Op.
  linalg::ReduceOp maxReduce;
  linalg::GenericOp exp1Op, exp2Op;
  for (Operation *u : S1.getUsers()) {
    if (u == scaleOp || isa<memref::DeallocOp>(u))
      continue;
    if (auto r = dyn_cast<linalg::ReduceOp>(u)) {
      if (maxReduce)
        return false;
      maxReduce = r;
    } else if (auto g = dyn_cast<linalg::GenericOp>(u)) {
      bool hasExp = false;
      g->walk([&](math::ExpOp) { hasExp = true; });
      if (!hasExp || g.getNumDpsInputs() != 2 || g.getNumDpsInits() != 1 ||
          g.getNumLoops() != 3 || g.getNumParallelLoops() != 3)
        return false;
      if (!exp1Op)
        exp1Op = g;
      else if (!exp2Op)
        exp2Op = g;
      else
        return false;
    } else {
      return false;
    }
  }
  if (!maxReduce || !exp1Op || !exp2Op)
    return false;
  // Disambiguate exp1Op vs exp2Op: exp1Op's output feeds sumReduce; exp2Op's
  // output feeds bmm2.
  auto getSingleNonDeallocLinalgUser = [&](Value v) -> Operation * {
    Operation *found = nullptr;
    for (Operation *u : v.getUsers()) {
      if (isa<memref::DeallocOp>(u) || u->getResultTypes().empty() == false)
        continue;
      if (u == exp1Op || u == exp2Op)
        continue;
      if (found)
        return nullptr;
      found = u;
    }
    return found;
  };
  (void)getSingleNonDeallocLinalgUser;
  auto hasReduceUser = [](Value v) {
    for (Operation *u : v.getUsers())
      if (isa<linalg::ReduceOp>(u))
        return true;
    return false;
  };
  if (hasReduceUser(exp2Op.getOutputs()[0]))
    std::swap(exp1Op, exp2Op);

  Value maxBuf = maxReduce.getInits()[0];
  auto maxAlloc = maxBuf.getDefiningOp<memref::AllocOp>();
  Operation *maxFillOp = nullptr;
  if (!maxAlloc || !stripToZeroInitOp(maxBuf, maxReduce, maxFillOp))
    return false;
  auto maxFill = dyn_cast_or_null<linalg::FillOp>(maxFillOp);
  if (!maxFill)
    return false;
  Operation *maxView = nullptr;
  for (Operation *u : maxBuf.getUsers()) {
    if (u == maxReduce || u == maxFill || isa<memref::DeallocOp>(u))
      continue;
    if (isPureViewOf(u, maxBuf) && !maxView)
      maxView = u;
    else
      return false;
  }
  if (!maxView)
    return false;
  Value maxViewVal = maxView->getResult(0);

  Value exp1Buf = exp1Op.getOutputs()[0];
  auto exp1Alloc = exp1Buf.getDefiningOp<memref::AllocOp>();
  if (!exp1Alloc || exp1Buf.getType() != s0Ty ||
      exp1Op.getInputs()[1] != maxViewVal)
    return false;

  linalg::ReduceOp sumReduce;
  for (Operation *u : exp1Buf.getUsers()) {
    if (u == exp1Op || isa<memref::DeallocOp>(u))
      continue;
    if (auto r = dyn_cast<linalg::ReduceOp>(u)) {
      if (sumReduce)
        return false;
      sumReduce = r;
    } else {
      return false;
    }
  }
  if (!sumReduce)
    return false;
  Value sumBuf = sumReduce.getInits()[0];
  auto sumAlloc = sumBuf.getDefiningOp<memref::AllocOp>();
  Operation *sumFillOp = nullptr;
  if (!sumAlloc || !stripToZeroInitOp(sumBuf, sumReduce, sumFillOp))
    return false;
  auto sumFill = dyn_cast_or_null<linalg::FillOp>(sumFillOp);
  if (!sumFill)
    return false;
  Operation *sumView = nullptr;
  for (Operation *u : sumBuf.getUsers()) {
    if (u == sumReduce || u == sumFill || isa<memref::DeallocOp>(u))
      continue;
    if (isPureViewOf(u, sumBuf) && !sumView)
      sumView = u;
    else
      return false;
  }
  if (!sumView)
    return false;
  Value sumViewVal = sumView->getResult(0);

  linalg::GenericOp logAddOp;
  for (Operation *u : sumViewVal.getUsers()) {
    if (auto g = dyn_cast<linalg::GenericOp>(u)) {
      if (logAddOp)
        return false;
      logAddOp = g;
    } else {
      return false;
    }
  }
  if (!logAddOp || logAddOp.getNumDpsInputs() != 2 ||
      logAddOp.getNumDpsInits() != 1 || logAddOp.getInputs()[0] != maxViewVal ||
      logAddOp.getInputs()[1] != sumViewVal)
    return false;
  Value logBuf = logAddOp.getOutputs()[0];
  auto logAlloc = logBuf.getDefiningOp<memref::AllocOp>();
  if (!logAlloc || exp2Op.getInputs()[1] != logBuf)
    return false;
  for (Operation *u : maxViewVal.getUsers())
    if (u != exp1Op && u != logAddOp)
      return false;
  for (Operation *u : logBuf.getUsers())
    if (u != logAddOp && u != exp2Op && !isa<memref::DeallocOp>(u))
      return false;

  Value P = exp2Op.getOutputs()[0];
  auto pAlloc = P.getDefiningOp<memref::AllocOp>();
  if (!pAlloc || P.getType() != s0Ty)
    return false;
  linalg::BatchMatmulOp bmm2;
  for (Operation *u : P.getUsers()) {
    if (u == exp2Op || isa<memref::DeallocOp>(u))
      continue;
    if (auto bm = dyn_cast<linalg::BatchMatmulOp>(u)) {
      if (bmm2)
        return false;
      bmm2 = bm;
    } else {
      return false;
    }
  }
  if (!bmm2 || bmm2->getBlock() != bmm1->getBlock() ||
      !isDefaultBmmMaps(bmm2) || bmm2.getNumDpsInputs() != 2 ||
      bmm2.getInputs()[0] != P || bmm2.getNumDpsInits() != 1)
    return false;
  Value V = bmm2.getInputs()[1], O = bmm2.getOutputs()[0];
  auto vTy = dyn_cast<MemRefType>(V.getType()),
       oTy = dyn_cast<MemRefType>(O.getType());
  if (!vTy || !oTy || !vTy.hasStaticShape() || !oTy.hasStaticShape() ||
      vTy.getRank() != 3 || oTy.getRank() != 3 || vTy != kTy || oTy != qTy ||
      !vTy.getLayout().isIdentity() || !oTy.getLayout().isIdentity())
    return false;
  Operation *bmm2FillOp = nullptr;
  for (Operation *u : O.getUsers()) {
    if (auto f = dyn_cast<linalg::FillOp>(u)) {
      if (f->getBlock() == bmm2->getBlock() && f->isBeforeInBlock(bmm2) &&
          f.getOutputs()[0] == O) {
        if (bmm2FillOp)
          return false;
        bmm2FillOp = f;
      }
    }
  }
  auto bmm2Fill = dyn_cast_or_null<linalg::FillOp>(bmm2FillOp);
  if (!bmm2Fill)
    return false;

  chain.bmm1 = bmm1;
  chain.bmm1Init = bmm1Init;
  chain.scaleOp = scaleOp;
  chain.maxFill = maxFill;
  chain.maxReduce = maxReduce;
  chain.maxView = maxView;
  chain.exp1Op = exp1Op;
  chain.sumFill = sumFill;
  chain.sumReduce = sumReduce;
  chain.sumView = sumView;
  chain.logAddOp = logAddOp;
  chain.exp2Op = exp2Op;
  chain.bmm2Fill = bmm2Fill;
  chain.bmm2 = bmm2;
  chain.Q = Q;
  chain.K = K;
  chain.V = V;
  chain.mask = mask;
  chain.O = O;
  chain.B = B;
  chain.M = M;
  chain.N = N;
  chain.D = D;
  chain.deadBuffers = {
      P, exp2Op.getOutputs()[0], logBuf, exp1Buf, sumBuf, maxBuf, S1, S0};
  // Deduplicate P/exp2Op.getOutputs()[0] in deadAllocs:
  SmallPtrSet<Operation *, 8> seenAllocs;
  for (Value buf : {S0, S1, maxBuf, exp1Buf, sumBuf, logBuf, P}) {
    auto alloc = buf.getDefiningOp<memref::AllocOp>();
    if (alloc && seenAllocs.insert(alloc).second) {
      chain.deadAllocs.push_back(alloc);
      for (Operation *u : buf.getUsers())
        if (auto dealloc = dyn_cast<memref::DeallocOp>(u))
          chain.deadDeallocs.push_back(dealloc);
    }
  }
  return true;
}

static void rewriteFlashAttentionBlocks(ModuleOp module, bool verifyFP16) {
  std::vector<FlashAttentionChain> chains;
  module.walk([&](linalg::BatchMatmulOp op) {
    FlashAttentionChain chain;
    if (matchFlashAttentionChain(op, chain))
      chains.push_back(std::move(chain));
  });
  IRRewriter rewriter(module.getContext());

  for (FlashAttentionChain &c : chains) {
    Location loc = c.bmm2.getLoc();
    rewriter.setInsertionPoint(c.bmm2);
    if (verifyFP16)
      insertFP16IMECheck(rewriter, loc, module, /*a100=*/true,
                         c.bmm2->getParentOfType<func::FuncOp>());

    Type f16 = rewriter.getF16Type();
    Type f32 = rewriter.getF32Type();
    auto index = [&](int64_t v) -> Value {
      return arith::ConstantIndexOp::create(rewriter, loc, v);
    };
    Value c0 = index(0), c1 = index(1), c8 = index(8), c32 = index(32);
    Value zeroF16 = arith::ConstantOp::create(rewriter, loc, f16,
                                              rewriter.getFloatAttr(f16, 0.0));
    Value zeroF32 = arith::ConstantOp::create(rewriter, loc, f32,
                                              rewriter.getFloatAttr(f32, 0.0));
    Value oneF32 = arith::ConstantOp::create(rewriter, loc, f32,
                                             rewriter.getFloatAttr(f32, 1.0));
    Value negInfF32 = arith::ConstantOp::create(
        rewriter, loc, f32, rewriter.getFloatAttr(f32, -1.0e30));
    auto vec8hTy = VectorType::get({8}, f16);
    auto vec32hTy = VectorType::get({32}, f16);
    auto vec32fTy = VectorType::get({32}, f32);
    Value zeroVec8h =
        vector::BroadcastOp::create(rewriter, loc, vec8hTy, zeroF16);
    Value zeroVec32h =
        vector::BroadcastOp::create(rewriter, loc, vec32hTy, zeroF16);
    Value zeroVec32f =
        vector::BroadcastOp::create(rewriter, loc, vec32fTy, zeroF32);

    int64_t B = c.B, M = c.M, N = c.N, D = c.D;
    int64_t paddedM8 = ((M + 7) / 8) * 8, numQ = paddedM8 / 8;
    int64_t paddedN8 = ((N + 7) / 8) * 8;
    int64_t paddedN32 = ((N + 31) / 32) * 32, numKP = paddedN32 / 32;
    int64_t paddedD32 = ((D + 31) / 32) * 32, numVP = paddedD32 / 32;
    int64_t tiledN32 = N / 32;
    int64_t completeN8 = (N / 8) * 8;
    int64_t tiledD32 = D / 32;

    Value vB = index(B), vM = index(M), vN = index(N), vD = index(D);
    Value vNumQ = index(numQ), vNumKP = index(numKP), vNumVP = index(numVP);
    Value vPaddedN8 = index(paddedN8), vPaddedN32 = index(paddedN32),
          vPaddedD32 = index(paddedD32);
    Value kPanelStride = index(D * 32);
    Value vPanelStride = index(paddedN8 * 32);

    SmallVector<ReassociationIndices> reassocBN = {{0, 1}, {2}};
    Value kFlat =
        memref::CollapseShapeOp::create(rewriter, loc, c.K, reassocBN);
    Value vFlat =
        memref::CollapseShapeOp::create(rewriter, loc, c.V, reassocBN);

    Value packedK = memref::AllocOp::create(
        rewriter, loc, MemRefType::get({B * numKP * D * 32}, f16));
    Value packedV = memref::AllocOp::create(
        rewriter, loc, MemRefType::get({B * numVP * paddedN8 * 32}, f16));

    OpFoldResult idx0 = rewriter.getIndexAttr(0);
    OpFoldResult idx1 = rewriter.getIndexAttr(1);

    // Pre-pack K^T [D, N] into [B, numKP, D/8, 32, 8].
    if (tiledN32 > 0) {
      Value vTiledKP = index(tiledN32);
      auto packKLoop =
          scf::ParallelOp::create(rewriter, loc, ValueRange{c0, c0},
                                  ValueRange{vB, vTiledKP}, ValueRange{c1, c1});
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(packKLoop.getBody());
      Value b = packKLoop.getInductionVars()[0];
      Value p = packKLoop.getInductionVars()[1];
      Value bOff = arith::MulIOp::create(rewriter, loc, b, vN);
      Value kSlice = memref::SubViewOp::create(
          rewriter, loc, kFlat, SmallVector<OpFoldResult>{bOff, idx0},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(N),
                                    rewriter.getIndexAttr(D)},
          SmallVector<OpFoldResult>{idx1, idx1});
      Value ktView = createZeroCopyTransposedView(rewriter, loc, kSlice);
      Value panelIdx = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, b, vNumKP), p);
      Value panelOff =
          arith::MulIOp::create(rewriter, loc, panelIdx, kPanelStride);
      Value kPanel = memref::SubViewOp::create(
          rewriter, loc, packedK, SmallVector<OpFoldResult>{panelOff},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(D * 32)},
          SmallVector<OpFoldResult>{idx1});
      Value col = arith::MulIOp::create(rewriter, loc, p, c32);
      PackBF16Op::create(rewriter, loc, kPanel, ktView, col,
                         rewriter.getI64IntegerAttr(D));
    }
    if (N % 32 != 0) {
      Value vColBase = index(tiledN32 * 32);
      Value vRemN = index(N - tiledN32 * 32);
      Value vTailKPIdx = index(numKP - 1);
      auto tailKLoop = scf::ParallelOp::create(rewriter, loc, ValueRange{c0},
                                               ValueRange{vB}, ValueRange{c1});
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(tailKLoop.getBody());
      Value b = tailKLoop.getInductionVars()[0];
      Value panelIdx = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, b, vNumKP),
          vTailKPIdx);
      Value panelOff =
          arith::MulIOp::create(rewriter, loc, panelIdx, kPanelStride);
      Value kPanel = memref::SubViewOp::create(
          rewriter, loc, packedK, SmallVector<OpFoldResult>{panelOff},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(D * 32)},
          SmallVector<OpFoldResult>{idx1});
      emitZeroFill1DF16(rewriter, loc, kPanel, D * 32);
      auto cLoop = scf::ForOp::create(rewriter, loc, c0, vRemN, c1);
      rewriter.setInsertionPointToStart(cLoop.getBody());
      Value cc = cLoop.getInductionVar();
      Value nIdx = arith::AddIOp::create(rewriter, loc, vColBase, cc);
      Value cOff = arith::MulIOp::create(rewriter, loc, cc, c8);
      auto kbLoop = scf::ForOp::create(rewriter, loc, c0, vD, c8);
      rewriter.setInsertionPointToStart(kbLoop.getBody());
      Value kb = kbLoop.getInductionVar();
      Value dstOff = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, kb, c32), cOff);
      Value vec = vector::LoadOp::create(rewriter, loc, vec8hTy, c.K,
                                         ValueRange{b, nIdx, kb});
      vector::StoreOp::create(rewriter, loc, vec, kPanel, ValueRange{dstOff});
    }

    // Pre-pack V [N, D] into [B, numVP, paddedN8/8, 32, 8].
    if (tiledD32 > 0) {
      Value vTiledVP = index(tiledD32);
      auto packVLoop =
          scf::ParallelOp::create(rewriter, loc, ValueRange{c0, c0},
                                  ValueRange{vB, vTiledVP}, ValueRange{c1, c1});
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(packVLoop.getBody());
      Value b = packVLoop.getInductionVars()[0];
      Value p = packVLoop.getInductionVars()[1];
      Value panelIdx = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, b, vNumVP), p);
      Value panelOff =
          arith::MulIOp::create(rewriter, loc, panelIdx, vPanelStride);
      Value vPanel = memref::SubViewOp::create(
          rewriter, loc, packedV, SmallVector<OpFoldResult>{panelOff},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(paddedN8 * 32)},
          SmallVector<OpFoldResult>{idx1});
      Value col = arith::MulIOp::create(rewriter, loc, p, c32);
      if (completeN8 > 0) {
        Value bOff = arith::MulIOp::create(rewriter, loc, b, vN);
        Value vSlice = memref::SubViewOp::create(
            rewriter, loc, vFlat, SmallVector<OpFoldResult>{bOff, idx0},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(completeN8),
                                      rewriter.getIndexAttr(D)},
            SmallVector<OpFoldResult>{idx1, idx1});
        Value subPanel = memref::SubViewOp::create(
            rewriter, loc, vPanel, SmallVector<OpFoldResult>{idx0},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(completeN8 * 32)},
            SmallVector<OpFoldResult>{idx1});
        PackBF16Op::create(rewriter, loc, subPanel, vSlice, col,
                           rewriter.getI64IntegerAttr(completeN8));
      }
      if (completeN8 < N) {
        Value tailOff = index(completeN8 * 32);
        Value tailPanel = memref::SubViewOp::create(
            rewriter, loc, vPanel, SmallVector<OpFoldResult>{tailOff},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(8 * 32)},
            SmallVector<OpFoldResult>{idx1});
        emitZeroFill1DF16(rewriter, loc, tailPanel, 8 * 32);
        Value vCompN = index(completeN8);
        Value remN = index(N - completeN8);
        auto rLoop = scf::ForOp::create(rewriter, loc, c0, remN, c1);
        rewriter.setInsertionPointToStart(rLoop.getBody());
        Value r = rLoop.getInductionVar();
        Value nIdx = arith::AddIOp::create(rewriter, loc, vCompN, r);
        auto ccLoop = scf::ForOp::create(rewriter, loc, c0, c32, c1);
        rewriter.setInsertionPointToStart(ccLoop.getBody());
        Value cc = ccLoop.getInductionVar();
        Value dIdx = arith::AddIOp::create(rewriter, loc, col, cc);
        Value dstOff = arith::AddIOp::create(
            rewriter, loc, arith::MulIOp::create(rewriter, loc, cc, c8), r);
        Value val = memref::LoadOp::create(rewriter, loc, c.V,
                                           ValueRange{b, nIdx, dIdx});
        memref::StoreOp::create(rewriter, loc, val, tailPanel,
                                ValueRange{dstOff});
      }
    }
    if (D % 32 != 0) {
      Value vColBase = index(tiledD32 * 32);
      Value vRemD = index(D - tiledD32 * 32);
      Value vTailVPIdx = index(numVP - 1);
      auto tailVLoop = scf::ParallelOp::create(rewriter, loc, ValueRange{c0},
                                               ValueRange{vB}, ValueRange{c1});
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(tailVLoop.getBody());
      Value b = tailVLoop.getInductionVars()[0];
      Value panelIdx = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, b, vNumVP),
          vTailVPIdx);
      Value panelOff =
          arith::MulIOp::create(rewriter, loc, panelIdx, vPanelStride);
      Value vPanel = memref::SubViewOp::create(
          rewriter, loc, packedV, SmallVector<OpFoldResult>{panelOff},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(paddedN8 * 32)},
          SmallVector<OpFoldResult>{idx1});
      emitZeroFill1DF16(rewriter, loc, vPanel, paddedN8 * 32);
      Value vCompN = index(completeN8);
      if (completeN8 > 0) {
        auto kbLoop = scf::ForOp::create(rewriter, loc, c0, vCompN, c8);
        rewriter.setInsertionPointToStart(kbLoop.getBody());
        Value kb = kbLoop.getInductionVar();
        Value blkBase = arith::MulIOp::create(rewriter, loc, kb, c32);
        auto rLoop = scf::ForOp::create(rewriter, loc, c0, c8, c1);
        rewriter.setInsertionPointToStart(rLoop.getBody());
        Value r = rLoop.getInductionVar();
        Value nIdx = arith::AddIOp::create(rewriter, loc, kb, r);
        Value rOff = arith::AddIOp::create(rewriter, loc, blkBase, r);
        auto ccLoop = scf::ForOp::create(rewriter, loc, c0, vRemD, c1);
        rewriter.setInsertionPointToStart(ccLoop.getBody());
        Value cc = ccLoop.getInductionVar();
        Value dIdx = arith::AddIOp::create(rewriter, loc, vColBase, cc);
        Value dstOff = arith::AddIOp::create(
            rewriter, loc, rOff, arith::MulIOp::create(rewriter, loc, cc, c8));
        Value val = memref::LoadOp::create(rewriter, loc, c.V,
                                           ValueRange{b, nIdx, dIdx});
        memref::StoreOp::create(rewriter, loc, val, vPanel, ValueRange{dstOff});
      }
      if (completeN8 < N) {
        rewriter.setInsertionPoint(tailVLoop.getBody()->getTerminator());
        Value remN = index(N - completeN8);
        Value blkBase = index(completeN8 * 32);
        auto rLoop = scf::ForOp::create(rewriter, loc, c0, remN, c1);
        rewriter.setInsertionPointToStart(rLoop.getBody());
        Value r = rLoop.getInductionVar();
        Value nIdx = arith::AddIOp::create(rewriter, loc, vCompN, r);
        Value rOff = arith::AddIOp::create(rewriter, loc, blkBase, r);
        auto ccLoop = scf::ForOp::create(rewriter, loc, c0, vRemD, c1);
        rewriter.setInsertionPointToStart(ccLoop.getBody());
        Value cc = ccLoop.getInductionVar();
        Value dIdx = arith::AddIOp::create(rewriter, loc, vColBase, cc);
        Value dstOff = arith::AddIOp::create(
            rewriter, loc, rOff, arith::MulIOp::create(rewriter, loc, cc, c8));
        Value val = memref::LoadOp::create(rewriter, loc, c.V,
                                           ValueRange{b, nIdx, dIdx});
        memref::StoreOp::create(rewriter, loc, val, vPanel, ValueRange{dstOff});
      }
    }

    // 2D Parallel Flash-Attention Loop: Q_tile @ K^T -> Softmax(f32) -> P @ V
    auto faLoop =
        scf::ParallelOp::create(rewriter, loc, ValueRange{c0, c0},
                                ValueRange{vB, vNumQ}, ValueRange{c1, c1});
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(faLoop.getBody());
      Value b = faLoop.getInductionVars()[0];
      Value qTile = faLoop.getInductionVars()[1];
      Value qBase = arith::MulIOp::create(rewriter, loc, qTile, c8);

      auto scope = memref::AllocaScopeOp::create(rewriter, loc, TypeRange{});
      Block *sb = rewriter.createBlock(&scope.getBodyRegion());
      rewriter.setInsertionPointToStart(sb);

      Value packedQ = memref::AllocaOp::create(rewriter, loc,
                                               MemRefType::get({8 * D}, f16));
      Value sTile = memref::AllocaOp::create(
          rewriter, loc, MemRefType::get({8, paddedN32}, f16));
      Value sF32Row = memref::AllocaOp::create(
          rewriter, loc, MemRefType::get({paddedN32}, f32));
      Value packedP = memref::AllocaOp::create(
          rewriter, loc, MemRefType::get({8 * paddedN8}, f16));
      Value oTile = memref::AllocaOp::create(
          rewriter, loc, MemRefType::get({8, paddedD32}, f16));

      Value remQ = arith::SubIOp::create(rewriter, loc, vM, qBase);
      Value hasFull8 = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::sge, remQ, c8);
      Value validRows =
          arith::SelectOp::create(rewriter, loc, hasFull8, c8, remQ);

      // 1. Pack Q[b, qBase..qBase+8, 0..D] into packedQ.
      auto qPackR = scf::ForOp::create(rewriter, loc, c0, c8, c1);
      {
        OpBuilder::InsertionGuard gq(rewriter);
        rewriter.setInsertionPointToStart(qPackR.getBody());
        Value r = qPackR.getInductionVar();
        Value qIdx = arith::AddIOp::create(rewriter, loc, qBase, r);
        Value rOff = arith::MulIOp::create(rewriter, loc, r, c8);
        if (M % 8 == 0) {
          auto qPackK = scf::ForOp::create(rewriter, loc, c0, vD, c8);
          rewriter.setInsertionPointToStart(qPackK.getBody());
          Value kb = qPackK.getInductionVar();
          Value dstOff = arith::AddIOp::create(
              rewriter, loc, arith::MulIOp::create(rewriter, loc, kb, c8),
              rOff);
          Value vec = vector::LoadOp::create(rewriter, loc, vec8hTy, c.Q,
                                             ValueRange{b, qIdx, kb});
          vector::StoreOp::create(rewriter, loc, vec, packedQ,
                                  ValueRange{dstOff});
        } else {
          Value inBound = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::ult, r, validRows);
          auto ifOp = scf::IfOp::create(rewriter, loc, inBound, true);
          rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
          auto qPackK = scf::ForOp::create(rewriter, loc, c0, vD, c8);
          rewriter.setInsertionPointToStart(qPackK.getBody());
          Value kb = qPackK.getInductionVar();
          Value dstOff = arith::AddIOp::create(
              rewriter, loc, arith::MulIOp::create(rewriter, loc, kb, c8),
              rOff);
          Value vec = vector::LoadOp::create(rewriter, loc, vec8hTy, c.Q,
                                             ValueRange{b, qIdx, kb});
          vector::StoreOp::create(rewriter, loc, vec, packedQ,
                                  ValueRange{dstOff});

          rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
          auto qZeroK = scf::ForOp::create(rewriter, loc, c0, vD, c8);
          rewriter.setInsertionPointToStart(qZeroK.getBody());
          Value kbZ = qZeroK.getInductionVar();
          Value dstOffZ = arith::AddIOp::create(
              rewriter, loc, arith::MulIOp::create(rewriter, loc, kbZ, c8),
              rOff);
          vector::StoreOp::create(rewriter, loc, zeroVec8h, packedQ,
                                  ValueRange{dstOffZ});
        }
      }

      // 2. Zero sTile [8, paddedN32] and compute S = Q_tile @ K_b^T.
      {
        auto zrLoop = scf::ForOp::create(rewriter, loc, c0, c8, c1);
        OpBuilder::InsertionGuard gz(rewriter);
        rewriter.setInsertionPointToStart(zrLoop.getBody());
        Value r = zrLoop.getInductionVar();
        auto zcLoop = scf::ForOp::create(rewriter, loc, c0, vPaddedN32, c32);
        rewriter.setInsertionPointToStart(zcLoop.getBody());
        vector::StoreOp::create(rewriter, loc, zeroVec32h, sTile,
                                ValueRange{r, zcLoop.getInductionVar()});
      }
      {
        auto kpLoop = scf::ForOp::create(rewriter, loc, c0, vNumKP, c1);
        OpBuilder::InsertionGuard gk(rewriter);
        rewriter.setInsertionPointToStart(kpLoop.getBody());
        Value p = kpLoop.getInductionVar();
        Value panelIdx = arith::AddIOp::create(
            rewriter, loc, arith::MulIOp::create(rewriter, loc, b, vNumKP), p);
        Value panelOff =
            arith::MulIOp::create(rewriter, loc, panelIdx, kPanelStride);
        Value kPanel = memref::SubViewOp::create(
            rewriter, loc, packedK, SmallVector<OpFoldResult>{panelOff},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(D * 32)},
            SmallVector<OpFoldResult>{idx1});
        Value col = arith::MulIOp::create(rewriter, loc, p, c32);
        GemmF16Op::create(rewriter, loc, sTile, packedQ, kPanel, c0, col,
                          rewriter.getI64IntegerAttr(D),
                          rewriter.getI64IntegerAttr(paddedN32));
      }

      // 3. Softmax in L1 cache (FP32 max/exp/sum, vectorized math.exp).
      int64_t vecN = (N / 32) * 32;
      Value vVecN = index(vecN);
      {
        auto rLoop = scf::ForOp::create(rewriter, loc, c0, validRows, c1);
        OpBuilder::InsertionGuard gr(rewriter);
        rewriter.setInsertionPointToStart(rLoop.getBody());
        Value r = rLoop.getInductionVar();
        Value qIdx = arith::AddIOp::create(rewriter, loc, qBase, r);

        // Apply scaleOp body + extend to f32 + track row max.
        auto maxLoop = scf::ForOp::create(rewriter, loc, c0, vN, c1,
                                          ValueRange{negInfF32});
        {
          OpBuilder::InsertionGuard gm(rewriter);
          rewriter.setInsertionPointToStart(maxLoop.getBody());
          Value j = maxLoop.getInductionVar();
          Value sVal =
              memref::LoadOp::create(rewriter, loc, sTile, ValueRange{r, j});
          IRMapping mapper;
          Block *scaleBlock = c.scaleOp.getBody();
          mapper.map(scaleBlock->getArgument(0), sVal);
          if (c.mask) {
            Value mVal = memref::LoadOp::create(rewriter, loc, c.mask,
                                                ValueRange{c0, qIdx, j});
            mapper.map(scaleBlock->getArgument(1), mVal);
          }
          for (Operation &bodyOp : scaleBlock->without_terminator())
            rewriter.clone(bodyOp, mapper);
          auto yieldOp = cast<linalg::YieldOp>(scaleBlock->getTerminator());
          Value scaledF16 = mapper.lookupOrDefault(yieldOp.getOperand(0));
          Value sF32 = arith::ExtFOp::create(rewriter, loc, f32, scaledF16);
          memref::StoreOp::create(rewriter, loc, sF32, sF32Row, ValueRange{j});
          Value nextMax = arith::MaximumFOp::create(
              rewriter, loc, maxLoop.getRegionIterArgs()[0], sF32);
          scf::YieldOp::create(rewriter, loc, nextMax);
        }
        Value maxVal = maxLoop.getResult(0);
        Value sumVal = zeroF32;
        if (vecN > 0) {
          Value bMaxVec =
              vector::BroadcastOp::create(rewriter, loc, vec32fTy, maxVal);
          auto expVecLoop = scf::ForOp::create(rewriter, loc, c0, vVecN, c32,
                                               ValueRange{zeroVec32f});
          {
            OpBuilder::InsertionGuard ge(rewriter);
            rewriter.setInsertionPointToStart(expVecLoop.getBody());
            Value j = expVecLoop.getInductionVar();
            Value sVec = vector::LoadOp::create(rewriter, loc, vec32fTy,
                                                sF32Row, ValueRange{j});
            Value diff = arith::SubFOp::create(rewriter, loc, sVec, bMaxVec);
            Value eVec = math::ExpOp::create(rewriter, loc, diff);
            vector::StoreOp::create(rewriter, loc, eVec, sF32Row,
                                    ValueRange{j});
            Value acc = arith::AddFOp::create(
                rewriter, loc, expVecLoop.getRegionIterArgs()[0], eVec);
            scf::YieldOp::create(rewriter, loc, acc);
          }
          sumVal = vector::ReductionOp::create(rewriter, loc,
                                               vector::CombiningKind::ADD,
                                               expVecLoop.getResult(0));
        }
        if (vecN < N) {
          auto expTailLoop = scf::ForOp::create(rewriter, loc, vVecN, vN, c1,
                                                ValueRange{sumVal});
          {
            OpBuilder::InsertionGuard gt(rewriter);
            rewriter.setInsertionPointToStart(expTailLoop.getBody());
            Value j = expTailLoop.getInductionVar();
            Value sVal =
                memref::LoadOp::create(rewriter, loc, sF32Row, ValueRange{j});
            Value diff = arith::SubFOp::create(rewriter, loc, sVal, maxVal);
            Value eVal = math::ExpOp::create(rewriter, loc, diff);
            memref::StoreOp::create(rewriter, loc, eVal, sF32Row,
                                    ValueRange{j});
            Value nextSum = arith::AddFOp::create(
                rewriter, loc, expTailLoop.getRegionIterArgs()[0], eVal);
            scf::YieldOp::create(rewriter, loc, nextSum);
          }
          sumVal = expTailLoop.getResult(0);
        }

        Value invSum = arith::DivFOp::create(rewriter, loc, oneF32, sumVal);
        if (vecN > 0) {
          Value bInvVec =
              vector::BroadcastOp::create(rewriter, loc, vec32fTy, invSum);
          auto normVec = scf::ForOp::create(rewriter, loc, c0, vVecN, c32);
          OpBuilder::InsertionGuard gn(rewriter);
          rewriter.setInsertionPointToStart(normVec.getBody());
          Value j = normVec.getInductionVar();
          Value eVec = vector::LoadOp::create(rewriter, loc, vec32fTy, sF32Row,
                                              ValueRange{j});
          Value pVecF32 = arith::MulFOp::create(rewriter, loc, eVec, bInvVec);
          Value pVecF16 =
              arith::TruncFOp::create(rewriter, loc, vec32hTy, pVecF32);
          vector::StoreOp::create(rewriter, loc, pVecF16, sTile,
                                  ValueRange{r, j});
        }
        if (vecN < N) {
          auto normTail = scf::ForOp::create(rewriter, loc, vVecN, vN, c1);
          OpBuilder::InsertionGuard gn(rewriter);
          rewriter.setInsertionPointToStart(normTail.getBody());
          Value j = normTail.getInductionVar();
          Value eVal =
              memref::LoadOp::create(rewriter, loc, sF32Row, ValueRange{j});
          Value pF32 = arith::MulFOp::create(rewriter, loc, eVal, invSum);
          Value pF16 = arith::TruncFOp::create(rewriter, loc, f16, pF32);
          memref::StoreOp::create(rewriter, loc, pF16, sTile, ValueRange{r, j});
        }
        if (N < paddedN8) {
          auto zeroColTail =
              scf::ForOp::create(rewriter, loc, vN, vPaddedN8, c1);
          OpBuilder::InsertionGuard gn(rewriter);
          rewriter.setInsertionPointToStart(zeroColTail.getBody());
          memref::StoreOp::create(rewriter, loc, zeroF16, sTile,
                                  ValueRange{r, zeroColTail.getInductionVar()});
        }
      }
      if (M % 8 != 0) {
        auto zeroRowTail = scf::ForOp::create(rewriter, loc, validRows, c8, c1);
        OpBuilder::InsertionGuard gz(rewriter);
        rewriter.setInsertionPointToStart(zeroRowTail.getBody());
        Value r = zeroRowTail.getInductionVar();
        auto zeroCol = scf::ForOp::create(rewriter, loc, c0, vPaddedN8, c8);
        rewriter.setInsertionPointToStart(zeroCol.getBody());
        vector::StoreOp::create(rewriter, loc, zeroVec8h, sTile,
                                ValueRange{r, zeroCol.getInductionVar()});
      }

      // 4. Pack sTile[0..8, 0..paddedN8] -> packedP, zero oTile, compute P @ V.
      {
        auto nbLoop = scf::ForOp::create(rewriter, loc, c0, vPaddedN8, c8);
        OpBuilder::InsertionGuard gp(rewriter);
        rewriter.setInsertionPointToStart(nbLoop.getBody());
        Value nb = nbLoop.getInductionVar();
        Value nbOff = arith::MulIOp::create(rewriter, loc, nb, c8);
        auto rLoop = scf::ForOp::create(rewriter, loc, c0, c8, c1);
        rewriter.setInsertionPointToStart(rLoop.getBody());
        Value r = rLoop.getInductionVar();
        Value dstOff = arith::AddIOp::create(
            rewriter, loc, nbOff, arith::MulIOp::create(rewriter, loc, r, c8));
        Value vec = vector::LoadOp::create(rewriter, loc, vec8hTy, sTile,
                                           ValueRange{r, nb});
        vector::StoreOp::create(rewriter, loc, vec, packedP,
                                ValueRange{dstOff});
      }
      {
        auto zrLoop = scf::ForOp::create(rewriter, loc, c0, c8, c1);
        OpBuilder::InsertionGuard gz(rewriter);
        rewriter.setInsertionPointToStart(zrLoop.getBody());
        Value r = zrLoop.getInductionVar();
        auto zcLoop = scf::ForOp::create(rewriter, loc, c0, vPaddedD32, c32);
        rewriter.setInsertionPointToStart(zcLoop.getBody());
        vector::StoreOp::create(rewriter, loc, zeroVec32h, oTile,
                                ValueRange{r, zcLoop.getInductionVar()});
      }
      {
        auto vpLoop = scf::ForOp::create(rewriter, loc, c0, vNumVP, c1);
        OpBuilder::InsertionGuard gv(rewriter);
        rewriter.setInsertionPointToStart(vpLoop.getBody());
        Value p = vpLoop.getInductionVar();
        Value panelIdx = arith::AddIOp::create(
            rewriter, loc, arith::MulIOp::create(rewriter, loc, b, vNumVP), p);
        Value panelOff =
            arith::MulIOp::create(rewriter, loc, panelIdx, vPanelStride);
        Value vPanel = memref::SubViewOp::create(
            rewriter, loc, packedV, SmallVector<OpFoldResult>{panelOff},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(paddedN8 * 32)},
            SmallVector<OpFoldResult>{idx1});
        Value col = arith::MulIOp::create(rewriter, loc, p, c32);
        GemmF16Op::create(rewriter, loc, oTile, packedP, vPanel, c0, col,
                          rewriter.getI64IntegerAttr(paddedN8),
                          rewriter.getI64IntegerAttr(paddedD32));
      }

      // 5. Store oTile[0..validRows, 0..D] -> O[b, qBase + r, d].
      {
        auto rLoop = scf::ForOp::create(rewriter, loc, c0, validRows, c1);
        OpBuilder::InsertionGuard go(rewriter);
        rewriter.setInsertionPointToStart(rLoop.getBody());
        Value r = rLoop.getInductionVar();
        Value qIdx = arith::AddIOp::create(rewriter, loc, qBase, r);
        auto dLoop = scf::ForOp::create(rewriter, loc, c0, vD, c8);
        rewriter.setInsertionPointToStart(dLoop.getBody());
        Value d = dLoop.getInductionVar();
        Value vec = vector::LoadOp::create(rewriter, loc, vec8hTy, oTile,
                                           ValueRange{r, d});
        vector::StoreOp::create(rewriter, loc, vec, c.O,
                                ValueRange{b, qIdx, d});
      }

      rewriter.setInsertionPointToEnd(sb);
      memref::AllocaScopeReturnOp::create(rewriter, loc, ValueRange{});
    }

    memref::DeallocOp::create(rewriter, loc, packedV);
    memref::DeallocOp::create(rewriter, loc, packedK);

    // Erase the materialized Attention intermediate ops & heap allocations.
    SmallVector<Operation *> toErase = {
        c.bmm2,    c.bmm2Fill, c.exp2Op, c.logAddOp,  c.sumReduce,
        c.sumFill, c.sumView,  c.exp1Op, c.maxReduce, c.maxFill,
        c.maxView, c.scaleOp,  c.bmm1,   c.bmm1Init};
    for (Operation *op : toErase)
      if (op)
        rewriter.eraseOp(op);
    for (memref::DeallocOp dealloc : c.deadDeallocs) {
      Operation *parent = dealloc->getParentOp();
      rewriter.eraseOp(dealloc);
      if (auto ifOp = dyn_cast_or_null<scf::IfOp>(parent))
        if (ifOp.getNumResults() == 0 && ifOp.getElseRegion().empty() &&
            ifOp.thenBlock()->without_terminator().empty())
          rewriter.eraseOp(ifOp);
    }
    for (memref::AllocOp alloc : c.deadAllocs)
      if (alloc && alloc->use_empty())
        rewriter.eraseOp(alloc);
  }
}

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
  if (a100FP16 && a100Gemm && matmulShapes.empty() && packedVectorSize == 0) {
    rewritePatchEmbeddingConvs(module, verifyFP16);
    rewriteFlashAttentionBlocks(module, verifyFP16);
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
