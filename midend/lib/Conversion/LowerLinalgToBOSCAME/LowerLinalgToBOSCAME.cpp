//=== LowerLinalgToBOSCAME.cpp - Linalg to BOSCAME Dialect Lowering Pass --===//
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
// This file defines Linalg dialect lowering pass to BOSCAME dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"

using namespace mlir;
using namespace buddy::boscame;

namespace {

enum class ElementwiseKind {
  Add,
  Sub,
  Mul,
  MinS,
  MinU,
  MaxS,
  MaxU,
  FAdd,
  FSub,
  FMul,
  FMax,
};

enum class BroadcastKind {
  Row,
  Column,
  Element,
};

enum class BroadcastRegisterKind {
  TileA,
  TileB,
  Accumulation,
};

static unsigned getElementBitWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth();
  return 0;
}

static unsigned getElementByteWidth(Type type) {
  unsigned bitWidth = getElementBitWidth(type);
  return bitWidth == 0 ? 0 : (bitWidth + 7) / 8;
}

static LogicalResult getMsetTypeImm(Type elementType, int64_t &imm) {
  if (elementType.isInteger(4)) {
    imm = 4;
    return success();
  }
  if (elementType.isInteger(8)) {
    imm = 8;
    return success();
  }
  if (elementType.isInteger(16) || elementType.isF16() ||
      elementType.isBF16()) {
    imm = 16;
    return success();
  }
  if (elementType.isInteger(32) || elementType.isF32()) {
    imm = 32;
    return success();
  }
  if (elementType.isInteger(64) || elementType.isF64()) {
    imm = 64;
    return success();
  }
  return failure();
}

static bool isParallel(utils::IteratorType type) {
  return type == utils::IteratorType::parallel;
}

static bool isReduction(utils::IteratorType type) {
  return type == utils::IteratorType::reduction;
}

static bool hasPureBufferOperands(linalg::GenericOp op) {
  return op.hasPureBufferSemantics() && op.getNumDpsInits() == 1;
}

static bool isAffineMap(AffineMap map, unsigned numDims,
                        ArrayRef<unsigned> dimPositions) {
  if (map.getNumDims() != numDims || map.getNumSymbols() != 0 ||
      map.getNumResults() != dimPositions.size())
    return false;

  MLIRContext *ctx = map.getContext();
  for (auto [result, dimPos] : llvm::zip_equal(map.getResults(), dimPositions))
    if (result != getAffineDimExpr(dimPos, ctx))
      return false;
  return true;
}

static bool isIdentity2D(AffineMap map) {
  return isAffineMap(map, 2, ArrayRef<unsigned>{0, 1});
}

static bool isIdentityND(AffineMap map, unsigned rank) {
  SmallVector<unsigned> dims;
  dims.reserve(rank);
  for (unsigned i = 0; i < rank; ++i)
    dims.push_back(i);
  return isAffineMap(map, rank, dims);
}

static bool isGenericMatmulLike(linalg::GenericOp op) {
  if (!hasPureBufferOperands(op) || op.getNumDpsInputs() != 2)
    return false;

  auto iterators = op.getIteratorTypesArray();
  if (iterators.size() != 3 || !isParallel(iterators[0]) ||
      !isParallel(iterators[1]) || !isReduction(iterators[2]))
    return false;

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 3 || !isAffineMap(maps[0], 3, ArrayRef<unsigned>{0, 2}) ||
      !isAffineMap(maps[1], 3, ArrayRef<unsigned>{2, 1}) ||
      !isAffineMap(maps[2], 3, ArrayRef<unsigned>{0, 1}))
    return false;

  Block &body = op.getRegion().front();
  if (body.getNumArguments() != 3)
    return false;

  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return false;

  Operation *addOp = yieldOp.getValues()[0].getDefiningOp();
  if (!addOp)
    return false;

  StringRef addName = addOp->getName().getStringRef();
  if (addName != "arith.addi" && addName != "arith.addf")
    return false;

  Value lhs = addOp->getOperand(0);
  Value rhs = addOp->getOperand(1);
  Operation *mulOp = lhs.getDefiningOp();
  Value accumulator = rhs;
  if (!mulOp) {
    mulOp = rhs.getDefiningOp();
    accumulator = lhs;
  }
  if (!mulOp || accumulator != body.getArgument(2))
    return false;

  StringRef mulName = mulOp->getName().getStringRef();
  if (mulName != "arith.muli" && mulName != "arith.mulf")
    return false;

  return (mulOp->getOperand(0) == body.getArgument(0) &&
          mulOp->getOperand(1) == body.getArgument(1)) ||
         (mulOp->getOperand(0) == body.getArgument(1) &&
          mulOp->getOperand(1) == body.getArgument(0));
}

static bool matchElementwiseKind(linalg::GenericOp op, ElementwiseKind &kind) {
  if (!hasPureBufferOperands(op) || op.getNumDpsInputs() != 2)
    return false;

  auto iterators = op.getIteratorTypesArray();
  if (iterators.size() != 2 || !llvm::all_of(iterators, isParallel))
    return false;

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 3 || !llvm::all_of(maps, isIdentity2D))
    return false;

  Block &body = op.getRegion().front();
  if (body.getNumArguments() != 3)
    return false;

  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return false;

  Operation *opDef = yieldOp.getValues()[0].getDefiningOp();
  if (!opDef || opDef->getNumOperands() != 2)
    return false;

  bool directOrder = opDef->getOperand(0) == body.getArgument(0) &&
                     opDef->getOperand(1) == body.getArgument(1);
  bool swappedOrder = opDef->getOperand(0) == body.getArgument(1) &&
                      opDef->getOperand(1) == body.getArgument(0);
  if (!directOrder && !swappedOrder)
    return false;

  StringRef name = opDef->getName().getStringRef();
  if (name == "arith.addi")
    kind = ElementwiseKind::Add;
  else if (name == "arith.subi" && directOrder)
    kind = ElementwiseKind::Sub;
  else if (name == "arith.muli")
    kind = ElementwiseKind::Mul;
  else if (name == "arith.minsi")
    kind = ElementwiseKind::MinS;
  else if (name == "arith.minui")
    kind = ElementwiseKind::MinU;
  else if (name == "arith.maxsi")
    kind = ElementwiseKind::MaxS;
  else if (name == "arith.maxui")
    kind = ElementwiseKind::MaxU;
  else if (name == "arith.addf")
    kind = ElementwiseKind::FAdd;
  else if (name == "arith.subf" && directOrder)
    kind = ElementwiseKind::FSub;
  else if (name == "arith.mulf")
    kind = ElementwiseKind::FMul;
  else if (name == "arith.maximumf" || name == "arith.maxnumf")
    kind = ElementwiseKind::FMax;
  else
    return false;

  return true;
}

static bool matchUnarySquare(linalg::GenericOp op) {
  if (!hasPureBufferOperands(op) || op.getNumDpsInputs() != 1)
    return false;

  auto iterators = op.getIteratorTypesArray();
  if (iterators.size() < 2 || !llvm::all_of(iterators, isParallel))
    return false;

  unsigned rank = iterators.size();
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 2 || !isIdentityND(maps[0], rank) ||
      !isIdentityND(maps[1], rank))
    return false;

  Block &body = op.getRegion().front();
  if (body.getNumArguments() != 2)
    return false;

  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return false;

  Operation *opDef = yieldOp.getValues()[0].getDefiningOp();
  if (!opDef || opDef->getName().getStringRef() != "math.fpowi" ||
      opDef->getNumOperands() != 2 ||
      opDef->getOperand(0) != body.getArgument(0))
    return false;

  APInt exponent;
  if (!matchPattern(opDef->getOperand(1), m_ConstantInt(&exponent)))
    return false;
  return exponent.getSExtValue() == 2;
}

static bool matchTranspose(linalg::GenericOp op) {
  if (!hasPureBufferOperands(op) || op.getNumDpsInputs() != 1)
    return false;

  auto iterators = op.getIteratorTypesArray();
  if (iterators.size() != 2 || !llvm::all_of(iterators, isParallel))
    return false;

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 2 || !isAffineMap(maps[0], 2, ArrayRef<unsigned>{1, 0}) ||
      !isIdentity2D(maps[1]))
    return false;

  Block &body = op.getRegion().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  return yieldOp && yieldOp.getValues().size() == 1 &&
         yieldOp.getValues()[0] == body.getArgument(0);
}

static bool matchBroadcast(linalg::GenericOp op, BroadcastKind &kind) {
  if (!hasPureBufferOperands(op) || op.getNumDpsInputs() != 1)
    return false;

  auto iterators = op.getIteratorTypesArray();
  if (iterators.size() != 2 || !llvm::all_of(iterators, isParallel))
    return false;

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 2 || !isIdentity2D(maps[1]))
    return false;

  AffineMap inputMap = maps[0];
  if (inputMap.getNumDims() != 2 || inputMap.getNumSymbols() != 0)
    return false;

  if (inputMap.getNumResults() == 0) {
    kind = BroadcastKind::Element;
  } else if (isAffineMap(inputMap, 2, ArrayRef<unsigned>{1})) {
    kind = BroadcastKind::Row;
  } else if (isAffineMap(inputMap, 2, ArrayRef<unsigned>{0})) {
    kind = BroadcastKind::Column;
  } else {
    return false;
  }

  Block &body = op.getRegion().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  return yieldOp && yieldOp.getValues().size() == 1 &&
         yieldOp.getValues()[0] == body.getArgument(0);
}

static bool isLowerableGeneric(linalg::GenericOp op) {
  ElementwiseKind elementwiseKind;
  BroadcastKind broadcastKind;
  return isGenericMatmulLike(op) || matchElementwiseKind(op, elementwiseKind) ||
         matchUnarySquare(op) || matchTranspose(op) ||
         matchBroadcast(op, broadcastKind);
}

static Value createDim(PatternRewriter &rewriter, Location loc, Value memref,
                       int64_t dim) {
  return memref::DimOp::create(rewriter, loc, memref, dim);
}

static Value createIndexMin(PatternRewriter &rewriter, Location loc,
                            Value bound, Value iv, int64_t step) {
  Value remain = arith::SubIOp::create(rewriter, loc, bound, iv);
  Value stepVal = arith::ConstantIndexOp::create(rewriter, loc, step);
  Value cmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                    remain, stepVal);
  return arith::SelectOp::create(rewriter, loc, cmp, remain, stepVal);
}

static Value castIndexToI64(PatternRewriter &rewriter, Location loc,
                            Value value) {
  return arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(),
                                    value);
}

static Value createByteStride(PatternRewriter &rewriter, Location loc,
                              Value memref, unsigned dim = 0) {
  auto memrefType = cast<MemRefType>(memref.getType());
  unsigned rank = memrefType.getRank();
  unsigned bytesPerElem = getElementByteWidth(memrefType.getElementType());
  Value bytes = arith::ConstantIndexOp::create(rewriter, loc, bytesPerElem);

  if (rank == 0)
    return arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(),
                                      bytes);

  auto metadata =
      memref::ExtractStridedMetadataOp::create(rewriter, loc, memref);
  Value strideElem = metadata.getResult(2 + rank + dim);
  Value strideBytes = arith::MulIOp::create(rewriter, loc, strideElem, bytes);
  return arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(),
                                    strideBytes);
}

static Value createSubView(PatternRewriter &rewriter, Location loc,
                           Value source, ArrayRef<Value> offsets,
                           ArrayRef<Value> sizes) {
  SmallVector<OpFoldResult> offsetResults;
  SmallVector<OpFoldResult> sizeResults;
  for (Value offset : offsets)
    offsetResults.push_back(offset);
  for (Value size : sizes)
    sizeResults.push_back(size);
  SmallVector<OpFoldResult> strides(offsets.size(), rewriter.getIndexAttr(1));
  return memref::SubViewOp::create(rewriter, loc, source, offsetResults,
                                   sizeResults, strides);
}

static LogicalResult createMSetTypeAndTiles(PatternRewriter &rewriter,
                                            Operation *anchor, Location loc,
                                            Type elementType, Value currM,
                                            Value currN, Value currK = {}) {
  int64_t msetTypeImm = 0;
  if (failed(getMsetTypeImm(elementType, msetTypeImm)))
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME element type");

  MSettypeiOp::create(rewriter, loc, rewriter.getI64Type(), msetTypeImm);
  MSettilemOp::create(rewriter, loc, rewriter.getI64Type(),
                      castIndexToI64(rewriter, loc, currM));
  MSettilenOp::create(rewriter, loc, rewriter.getI64Type(),
                      castIndexToI64(rewriter, loc, currN));
  if (currK)
    MSettilekOp::create(rewriter, loc, rewriter.getI64Type(),
                        castIndexToI64(rewriter, loc, currK));
  return success();
}

static FailureOr<Value> createLoadA(PatternRewriter &rewriter,
                                    Operation *anchor, Location loc,
                                    Type elementType, Value source,
                                    Value stride) {
  Type tileType = VectorType::get({4, 4}, elementType);
  if (elementType.isInteger(8))
    return Mlae8mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(16) || elementType.isF16() ||
           elementType.isBF16())
    return Mlae16mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(32) || elementType.isF32())
    return Mlae32mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(64) || elementType.isF64())
    return Mlae64mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME A load type");
}

static FailureOr<Value> createLoadB(PatternRewriter &rewriter,
                                    Operation *anchor, Location loc,
                                    Type elementType, Value source,
                                    Value stride) {
  Type tileType = VectorType::get({4, 4}, elementType);
  if (elementType.isInteger(8))
    return Mlbe8mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(16) || elementType.isF16() ||
           elementType.isBF16())
    return Mlbe16mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(32) || elementType.isF32())
    return Mlbe32mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(64) || elementType.isF64())
    return Mlbe64mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME B load type");
}

static FailureOr<Value> createLoadC(PatternRewriter &rewriter,
                                    Operation *anchor, Location loc,
                                    Type elementType, Value source,
                                    Value stride) {
  Type tileType = VectorType::get({4, 4}, elementType);
  if (elementType.isInteger(8))
    return Mlce8mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(16) || elementType.isF16() ||
           elementType.isBF16())
    return Mlce16mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(32) || elementType.isF32())
    return Mlce32mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(64) || elementType.isF64())
    return Mlce64mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME C load type");
}

static LogicalResult createStoreC(PatternRewriter &rewriter, Operation *anchor,
                                  Location loc, Type elementType, Value src,
                                  Value dest, Value stride) {
  if (elementType.isInteger(8))
    Msce8mOp::create(rewriter, loc, src, dest, stride);
  else if (elementType.isInteger(16) || elementType.isF16() ||
           elementType.isBF16())
    Msce16mOp::create(rewriter, loc, src, dest, stride);
  else if (elementType.isInteger(32) || elementType.isF32())
    Msce32mOp::create(rewriter, loc, src, dest, stride);
  else if (elementType.isInteger(64) || elementType.isF64())
    Msce64mOp::create(rewriter, loc, src, dest, stride);
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME C store type");
  return success();
}

static FailureOr<Value> createMatmul(PatternRewriter &rewriter,
                                     Operation *anchor, Location loc,
                                     Type lhsType, Type resultType, Value acc,
                                     Value lhs, Value rhs) {
  Type tileType = VectorType::get({4, 4}, resultType);
  if (resultType.isInteger(32) && lhsType.isInteger(32))
    return MmaWmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (resultType.isInteger(32) && lhsType.isInteger(16))
    return MwmaHmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (resultType.isInteger(32) && lhsType.isInteger(8))
    return MqmaBmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (resultType.isInteger(16) && lhsType.isInteger(16))
    return MmaHmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (resultType.isInteger(64) && lhsType.isInteger(64))
    return MmaDwmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if ((lhsType.isF16() || lhsType.isBF16()) && resultType.isF32())
    return MfwmaHfmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (lhsType.isF32() && resultType.isF32())
    return MfmaFmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (lhsType.isF32() && resultType.isF64())
    return MfwmaFmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (lhsType.isF64() && resultType.isF64())
    return MfmaDmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else
    return rewriter.notifyMatchFailure(
        anchor, "unsupported BOSCAME matmul instruction type");
}

static Value createMatrixOp(PatternRewriter &rewriter, Location loc,
                            StringRef name, Type resultType,
                            ValueRange operands) {
  OperationState state(loc, name);
  state.addOperands(operands);
  state.addTypes(resultType);
  return rewriter.create(state)->getResult(0);
}

static FailureOr<Value> createElementwise(PatternRewriter &rewriter,
                                          Operation *anchor, Location loc,
                                          ElementwiseKind kind,
                                          Type elementType, Value lhs,
                                          Value rhs) {
  StringRef suffix;
  if (elementType.isInteger(8))
    suffix = ".b.mm";
  else if (elementType.isInteger(16))
    suffix = ".h.mm";
  else if (elementType.isInteger(32))
    suffix = ".w.mm";
  else if (elementType.isInteger(64))
    suffix = ".dw.mm";
  else if (elementType.isF16() || elementType.isBF16())
    suffix = ".hf.mm";
  else if (elementType.isF32())
    suffix = ".f.mm";
  else if (elementType.isF64())
    suffix = ".d.mm";
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME elementwise type");

  StringRef mnemonic;
  switch (kind) {
  case ElementwiseKind::Add:
    mnemonic = "madd";
    break;
  case ElementwiseKind::Sub:
    mnemonic = "msub";
    break;
  case ElementwiseKind::Mul:
    mnemonic = "mmul";
    break;
  case ElementwiseKind::MinS:
    mnemonic = "mmin";
    break;
  case ElementwiseKind::MinU:
    mnemonic = "mminu";
    break;
  case ElementwiseKind::MaxS:
    mnemonic = "mmax";
    break;
  case ElementwiseKind::MaxU:
    mnemonic = "mmaxu";
    break;
  case ElementwiseKind::FAdd:
    mnemonic = "mfadd";
    break;
  case ElementwiseKind::FSub:
    mnemonic = "mfsub";
    break;
  case ElementwiseKind::FMul:
    mnemonic = "mfmul";
    break;
  case ElementwiseKind::FMax:
    mnemonic = "mfmax";
    break;
  }
  std::string name = "bosc_ame." + mnemonic.str() + suffix.str();
  return createMatrixOp(rewriter, loc, name, lhs.getType(), {lhs, rhs});
}

static FailureOr<Value> createBroadcast(PatternRewriter &rewriter,
                                        Operation *anchor, Location loc,
                                        BroadcastKind kind,
                                        BroadcastRegisterKind regKind,
                                        Type elementType, Value src) {
  StringRef registerSuffix = regKind == BroadcastRegisterKind::TileA   ? "a"
                             : regKind == BroadcastRegisterKind::TileB ? "b"
                                                                       : "c";
  if (kind == BroadcastKind::Row) {
    std::string name = "bosc_ame.mbc" + registerSuffix.str() + "r.m";
    return createMatrixOp(rewriter, loc, name, src.getType(), {src});
  }

  StringRef bitWidth;
  if (elementType.isInteger(8))
    bitWidth = "8";
  else if (elementType.isInteger(16) || elementType.isF16() ||
           elementType.isBF16())
    bitWidth = "16";
  else if (elementType.isInteger(32) || elementType.isF32())
    bitWidth = "32";
  else if (elementType.isInteger(64) || elementType.isF64())
    bitWidth = "64";
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME broadcast type");

  std::string name = "bosc_ame.mbc" + registerSuffix.str() +
                     (kind == BroadcastKind::Column ? "ce" : "ee") +
                     bitWidth.str() + ".m";
  return createMatrixOp(rewriter, loc, name, src.getType(), {src});
}

static FailureOr<Value> createTranspose(PatternRewriter &rewriter,
                                        Operation *anchor, Location loc,
                                        BroadcastRegisterKind regKind,
                                        Type elementType, Value src) {
  StringRef registerSuffix = regKind == BroadcastRegisterKind::TileA   ? "a"
                             : regKind == BroadcastRegisterKind::TileB ? "b"
                                                                       : "c";
  StringRef bitWidth;
  if (elementType.isInteger(8))
    bitWidth = "8";
  else if (elementType.isInteger(16) || elementType.isF16() ||
           elementType.isBF16())
    bitWidth = "16";
  else if (elementType.isInteger(32) || elementType.isF32())
    bitWidth = "32";
  else if (elementType.isInteger(64) || elementType.isF64())
    bitWidth = "64";
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME transpose type");
  std::string name =
      "bosc_ame.mt" + registerSuffix.str() + "e" + bitWidth.str() + ".m";
  return createMatrixOp(rewriter, loc, name, src.getType(), {src});
}

static LogicalResult lowerMatmulLike(Operation *anchor,
                                     PatternRewriter &rewriter, Location loc,
                                     Value A, Value B, Value C) {
  auto AType = dyn_cast<MemRefType>(A.getType());
  auto BType = dyn_cast<MemRefType>(B.getType());
  auto CType = dyn_cast<MemRefType>(C.getType());
  if (!AType || !BType || !CType)
    return rewriter.notifyMatchFailure(anchor, "expected memref operands");

  Type elemTypeA = AType.getElementType();
  Type elemTypeB = BType.getElementType();
  Type elemTypeC = CType.getElementType();
  if (elemTypeA != elemTypeB)
    return rewriter.notifyMatchFailure(
        anchor, "matmul lhs and rhs element types differ");

  int64_t tileM = 4, tileN = 4, tileK = 4;
  if (elemTypeA.isInteger(8) && elemTypeC.isInteger(32))
    tileK = 16;
  else if ((elemTypeA.isF16() || elemTypeA.isBF16()) && elemTypeC.isF32())
    tileK = 8;
  else if (elemTypeA.isInteger(16) && elemTypeC.isInteger(32))
    tileK = 8;
  else if (elemTypeA.isInteger(32) && elemTypeC.isInteger(32))
    tileK = 4;
  else if (elemTypeA.isF32() && (elemTypeC.isF32() || elemTypeC.isF64()))
    tileK = 4;
  else if (elemTypeA.isInteger(64) && elemTypeC.isInteger(64))
    tileK = 2;
  else if (elemTypeA.isF64() && elemTypeC.isF64())
    tileK = 2;
  else
    return rewriter.notifyMatchFailure(anchor, "unsupported matmul precision");

  Value dimM = createDim(rewriter, loc, A, 0);
  Value dimK = createDim(rewriter, loc, A, 1);
  Value dimN = createDim(rewriter, loc, B, 1);

  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value stepM = arith::ConstantIndexOp::create(rewriter, loc, tileM);
  Value stepK = arith::ConstantIndexOp::create(rewriter, loc, tileK);
  Value stepN = arith::ConstantIndexOp::create(rewriter, loc, tileN);

  auto loopM = scf::ForOp::create(rewriter, loc, c0, dimM, stepM);
  rewriter.setInsertionPointToStart(loopM.getBody());
  Value ivM = loopM.getInductionVar();

  auto loopN = scf::ForOp::create(rewriter, loc, c0, dimN, stepN);
  rewriter.setInsertionPointToStart(loopN.getBody());
  Value ivN = loopN.getInductionVar();

  auto loopK = scf::ForOp::create(rewriter, loc, c0, dimK, stepK);
  rewriter.setInsertionPointToStart(loopK.getBody());
  Value ivK = loopK.getInductionVar();

  Value currM = createIndexMin(rewriter, loc, dimM, ivM, tileM);
  Value currN = createIndexMin(rewriter, loc, dimN, ivN, tileN);
  Value currK = createIndexMin(rewriter, loc, dimK, ivK, tileK);

  Value subA = createSubView(rewriter, loc, A, {ivM, ivK}, {currM, currK});
  Value subB = createSubView(rewriter, loc, B, {ivK, ivN}, {currK, currN});
  Value subC = createSubView(rewriter, loc, C, {ivM, ivN}, {currM, currN});

  Value strideA = createByteStride(rewriter, loc, subA);
  Value strideB = createByteStride(rewriter, loc, subB);
  Value strideC = createByteStride(rewriter, loc, subC);

  if (failed(createMSetTypeAndTiles(rewriter, anchor, loc, elemTypeA, currM,
                                    currN, currK)))
    return failure();
  FailureOr<Value> tileA =
      createLoadA(rewriter, anchor, loc, elemTypeA, subA, strideA);
  FailureOr<Value> tileB =
      createLoadB(rewriter, anchor, loc, elemTypeB, subB, strideB);
  FailureOr<Value> acc =
      createLoadC(rewriter, anchor, loc, elemTypeC, subC, strideC);
  if (failed(tileA) || failed(tileB) || failed(acc))
    return failure();
  FailureOr<Value> result = createMatmul(rewriter, anchor, loc, elemTypeA,
                                         elemTypeC, *acc, *tileA, *tileB);
  if (failed(result) || failed(createStoreC(rewriter, anchor, loc, elemTypeC,
                                            *result, subC, strideC)))
    return failure();

  rewriter.setInsertionPointAfter(loopM);
  return success();
}

class MatmulToBOSCAMELowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics())
      return failure();

    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value C = op.getDpsInitOperand(0)->get();

    if (failed(lowerMatmulLike(op, rewriter, op.getLoc(), A, B, C)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

class GenericMatmulToBOSCAMELowering
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!isGenericMatmulLike(op))
      return failure();

    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value C = op.getDpsInitOperand(0)->get();

    if (failed(lowerMatmulLike(op, rewriter, op.getLoc(), A, B, C)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

class GenericElementwiseToBOSCAMELowering
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    ElementwiseKind kind;
    if (!matchElementwiseKind(op, kind))
      return failure();

    Location loc = op.getLoc();
    Value lhs = op.getDpsInputOperand(0)->get();
    Value rhs = op.getDpsInputOperand(1)->get();
    Value out = op.getDpsInitOperand(0)->get();

    auto lhsType = dyn_cast<MemRefType>(lhs.getType());
    auto rhsType = dyn_cast<MemRefType>(rhs.getType());
    auto outType = dyn_cast<MemRefType>(out.getType());
    if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 ||
        rhsType.getRank() != 2 || outType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected rank-2 memrefs");

    Type elementType = outType.getElementType();
    if (lhsType.getElementType() != rhsType.getElementType() ||
        lhsType.getElementType() != elementType)
      return rewriter.notifyMatchFailure(op,
                                         "elementwise operand types differ");

    constexpr int64_t tileM = 4;
    constexpr int64_t tileN = 4;

    Value dimM = createDim(rewriter, loc, out, 0);
    Value dimN = createDim(rewriter, loc, out, 1);
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value stepM = arith::ConstantIndexOp::create(rewriter, loc, tileM);
    Value stepN = arith::ConstantIndexOp::create(rewriter, loc, tileN);

    auto loopM = scf::ForOp::create(rewriter, loc, c0, dimM, stepM);
    rewriter.setInsertionPointToStart(loopM.getBody());
    Value ivM = loopM.getInductionVar();

    auto loopN = scf::ForOp::create(rewriter, loc, c0, dimN, stepN);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    Value currM = createIndexMin(rewriter, loc, dimM, ivM, tileM);
    Value currN = createIndexMin(rewriter, loc, dimN, ivN, tileN);

    Value subLhs =
        createSubView(rewriter, loc, lhs, {ivM, ivN}, {currM, currN});
    Value subRhs =
        createSubView(rewriter, loc, rhs, {ivM, ivN}, {currM, currN});
    Value subOut =
        createSubView(rewriter, loc, out, {ivM, ivN}, {currM, currN});

    Value strideLhs = createByteStride(rewriter, loc, subLhs);
    Value strideRhs = createByteStride(rewriter, loc, subRhs);
    Value strideOut = createByteStride(rewriter, loc, subOut);

    if (failed(createMSetTypeAndTiles(rewriter, op, loc, elementType, currM,
                                      currN)))
      return failure();
    FailureOr<Value> lhsTile =
        createLoadC(rewriter, op, loc, elementType, subLhs, strideLhs);
    FailureOr<Value> rhsTile =
        createLoadC(rewriter, op, loc, elementType, subRhs, strideRhs);
    if (failed(lhsTile) || failed(rhsTile))
      return failure();
    FailureOr<Value> result = createElementwise(
        rewriter, op, loc, kind, elementType, *lhsTile, *rhsTile);
    if (failed(result) || failed(createStoreC(rewriter, op, loc, elementType,
                                              *result, subOut, strideOut)))
      return failure();

    rewriter.setInsertionPointAfter(loopM);
    rewriter.eraseOp(op);
    return success();
  }
};

class GenericUnarySquareToBOSCAMELowering
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!matchUnarySquare(op))
      return failure();

    Location loc = op.getLoc();
    Value input = op.getDpsInputOperand(0)->get();
    Value out = op.getDpsInitOperand(0)->get();

    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto outType = dyn_cast<MemRefType>(out.getType());
    if (!inputType || !outType || inputType.getRank() != outType.getRank() ||
        inputType.getRank() < 2)
      return rewriter.notifyMatchFailure(
          op, "expected same-rank memrefs with rank >= 2");

    if (inputType.getElementType() != outType.getElementType())
      return rewriter.notifyMatchFailure(op, "square operand types differ");

    Type elementType = outType.getElementType();
    if (!elementType.isF16() && !elementType.isBF16() && !elementType.isF32() &&
        !elementType.isF64())
      return rewriter.notifyMatchFailure(
          op, "math.fpowi square expects floating point element type");

    constexpr int64_t tileM = 4;
    constexpr int64_t tileN = 4;
    unsigned rank = outType.getRank();
    unsigned matrixDimM = rank - 2;
    unsigned matrixDimN = rank - 1;

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value stepM = arith::ConstantIndexOp::create(rewriter, loc, tileM);
    Value stepN = arith::ConstantIndexOp::create(rewriter, loc, tileN);

    SmallVector<Value> outerIvs;
    SmallVector<scf::ForOp> outerLoops;
    for (unsigned dim = 0; dim < matrixDimM; ++dim) {
      Value dimSize = createDim(rewriter, loc, out, dim);
      auto loop = scf::ForOp::create(rewriter, loc, c0, dimSize, c1);
      outerLoops.push_back(loop);
      rewriter.setInsertionPointToStart(loop.getBody());
      outerIvs.push_back(loop.getInductionVar());
    }

    Value dimM = createDim(rewriter, loc, out, matrixDimM);
    Value dimN = createDim(rewriter, loc, out, matrixDimN);

    auto loopM = scf::ForOp::create(rewriter, loc, c0, dimM, stepM);
    rewriter.setInsertionPointToStart(loopM.getBody());
    Value ivM = loopM.getInductionVar();

    auto loopN = scf::ForOp::create(rewriter, loc, c0, dimN, stepN);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    Value currM = createIndexMin(rewriter, loc, dimM, ivM, tileM);
    Value currN = createIndexMin(rewriter, loc, dimN, ivN, tileN);

    SmallVector<Value> offsets(outerIvs.begin(), outerIvs.end());
    offsets.push_back(ivM);
    offsets.push_back(ivN);

    SmallVector<Value> sizes(outerIvs.size(), c1);
    sizes.push_back(currM);
    sizes.push_back(currN);

    Value subInput = createSubView(rewriter, loc, input, offsets, sizes);
    Value subOut = createSubView(rewriter, loc, out, offsets, sizes);

    Value strideInput = createByteStride(rewriter, loc, subInput, matrixDimM);
    Value strideOut = createByteStride(rewriter, loc, subOut, matrixDimM);

    if (failed(createMSetTypeAndTiles(rewriter, op, loc, elementType, currM,
                                      currN)))
      return failure();
    FailureOr<Value> inputTile =
        createLoadC(rewriter, op, loc, elementType, subInput, strideInput);
    if (failed(inputTile))
      return failure();
    FailureOr<Value> result =
        createElementwise(rewriter, op, loc, ElementwiseKind::FMul, elementType,
                          *inputTile, *inputTile);
    if (failed(result) || failed(createStoreC(rewriter, op, loc, elementType,
                                              *result, subOut, strideOut)))
      return failure();

    if (!outerLoops.empty())
      rewriter.setInsertionPointAfter(outerLoops.front());
    else
      rewriter.setInsertionPointAfter(loopM);
    rewriter.eraseOp(op);
    return success();
  }
};

class GenericTransposeToBOSCAMELowering
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!matchTranspose(op))
      return failure();

    Location loc = op.getLoc();
    Value input = op.getDpsInputOperand(0)->get();
    Value out = op.getDpsInitOperand(0)->get();

    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto outType = dyn_cast<MemRefType>(out.getType());
    if (!inputType || !outType || inputType.getRank() != 2 ||
        outType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected rank-2 memrefs");

    Type elementType = outType.getElementType();
    if (inputType.getElementType() != elementType)
      return rewriter.notifyMatchFailure(op, "transpose element types differ");

    constexpr int64_t tileM = 4;
    constexpr int64_t tileN = 4;
    for (int64_t dim : outType.getShape()) {
      if (ShapedType::isDynamic(dim) || dim % tileM != 0)
        return rewriter.notifyMatchFailure(
            op, "BOSCAME transpose requires full 4x4 square tiles");
    }

    Value dimM = createDim(rewriter, loc, out, 0);
    Value dimN = createDim(rewriter, loc, out, 1);
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value stepM = arith::ConstantIndexOp::create(rewriter, loc, tileM);
    Value stepN = arith::ConstantIndexOp::create(rewriter, loc, tileN);

    auto loopM = scf::ForOp::create(rewriter, loc, c0, dimM, stepM);
    rewriter.setInsertionPointToStart(loopM.getBody());
    Value ivM = loopM.getInductionVar();

    auto loopN = scf::ForOp::create(rewriter, loc, c0, dimN, stepN);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    Value currM = createIndexMin(rewriter, loc, dimM, ivM, tileM);
    Value currN = createIndexMin(rewriter, loc, dimN, ivN, tileN);

    Value subInput =
        createSubView(rewriter, loc, input, {ivN, ivM}, {currN, currM});
    Value subOut =
        createSubView(rewriter, loc, out, {ivM, ivN}, {currM, currN});

    Value strideInput = createByteStride(rewriter, loc, subInput);
    Value strideOut = createByteStride(rewriter, loc, subOut);

    if (failed(createMSetTypeAndTiles(rewriter, op, loc, elementType, currM,
                                      currN)))
      return failure();
    FailureOr<Value> inputTile =
        createLoadC(rewriter, op, loc, elementType, subInput, strideInput);
    if (failed(inputTile))
      return failure();
    FailureOr<Value> result =
        createTranspose(rewriter, op, loc, BroadcastRegisterKind::Accumulation,
                        elementType, *inputTile);
    if (failed(result) || failed(createStoreC(rewriter, op, loc, elementType,
                                              *result, subOut, strideOut)))
      return failure();

    rewriter.setInsertionPointAfter(loopM);
    rewriter.eraseOp(op);
    return success();
  }
};

class GenericBroadcastToBOSCAMELowering
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    BroadcastKind kind;
    if (!matchBroadcast(op, kind))
      return failure();

    Location loc = op.getLoc();
    Value input = op.getDpsInputOperand(0)->get();
    Value out = op.getDpsInitOperand(0)->get();

    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto outType = dyn_cast<MemRefType>(out.getType());
    if (!inputType || !outType || outType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected memref operands");

    Type elementType = outType.getElementType();
    if (inputType.getElementType() != elementType)
      return rewriter.notifyMatchFailure(op, "broadcast element types differ");

    constexpr int64_t tileM = 4;
    constexpr int64_t tileN = 4;

    Value dimM = createDim(rewriter, loc, out, 0);
    Value dimN = createDim(rewriter, loc, out, 1);
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value stepM = arith::ConstantIndexOp::create(rewriter, loc, tileM);
    Value stepN = arith::ConstantIndexOp::create(rewriter, loc, tileN);

    auto loopM = scf::ForOp::create(rewriter, loc, c0, dimM, stepM);
    rewriter.setInsertionPointToStart(loopM.getBody());
    Value ivM = loopM.getInductionVar();

    auto loopN = scf::ForOp::create(rewriter, loc, c0, dimN, stepN);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    Value currM = createIndexMin(rewriter, loc, dimM, ivM, tileM);
    Value currN = createIndexMin(rewriter, loc, dimN, ivN, tileN);

    Value broadcastSource = input;
    if (kind == BroadcastKind::Row) {
      if (inputType.getRank() != 1)
        return rewriter.notifyMatchFailure(
            op, "row broadcast expects rank-1 input");
      broadcastSource = createSubView(rewriter, loc, input, {ivN}, {currN});
    } else if (kind == BroadcastKind::Column) {
      if (inputType.getRank() != 1)
        return rewriter.notifyMatchFailure(
            op, "column broadcast expects rank-1 input");
      broadcastSource = createSubView(rewriter, loc, input, {ivM}, {currM});
    } else if (inputType.getRank() != 0) {
      return rewriter.notifyMatchFailure(
          op, "element broadcast expects rank-0 input");
    }

    Value subOut =
        createSubView(rewriter, loc, out, {ivM, ivN}, {currM, currN});
    Value strideInput = createByteStride(rewriter, loc, broadcastSource);
    Value strideOut = createByteStride(rewriter, loc, subOut);

    if (failed(createMSetTypeAndTiles(rewriter, op, loc, elementType, currM,
                                      currN)))
      return failure();
    FailureOr<Value> inputTile = createLoadC(rewriter, op, loc, elementType,
                                             broadcastSource, strideInput);
    if (failed(inputTile))
      return failure();
    FailureOr<Value> result = createBroadcast(
        rewriter, op, loc, kind, BroadcastRegisterKind::Accumulation,
        elementType, *inputTile);
    if (failed(result) || failed(createStoreC(rewriter, op, loc, elementType,
                                              *result, subOut, strideOut)))
      return failure();

    rewriter.setInsertionPointAfter(loopM);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
class LowerLinalgToBOSCAMEPass
    : public PassWrapper<LowerLinalgToBOSCAMEPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToBOSCAMEPass)

  StringRef getArgument() const final { return "lower-linalg-to-boscame"; }
  StringRef getDescription() const final {
    return "Lower linalg dialect operations to BOSCAME dialect operations.";
  }

  LowerLinalgToBOSCAMEPass() = default;
  LowerLinalgToBOSCAMEPass(const LowerLinalgToBOSCAMEPass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<BOSCAMEDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LowerLinalgToBOSCAMEPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  RewritePatternSet patterns(context);
  patterns.add<
      MatmulToBOSCAMELowering, GenericMatmulToBOSCAMELowering,
      GenericElementwiseToBOSCAMELowering, GenericUnarySquareToBOSCAMELowering,
      GenericTransposeToBOSCAMELowering, GenericBroadcastToBOSCAMELowering>(
      context);

  ConversionTarget target(*context);
  target.addLegalDialect<BOSCAMEDialect, arith::ArithDialect,
                         memref::MemRefDialect, scf::SCFDialect>();
  target.addIllegalOp<linalg::MatmulOp>();
  target.addDynamicallyLegalOp<linalg::GenericOp>(
      [](linalg::GenericOp op) { return !isLowerableGeneric(op); });

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace buddy {
void registerLowerLinalgToBOSCAMEPass() {
  PassRegistration<LowerLinalgToBOSCAMEPass>();
}
} // namespace buddy
} // namespace mlir
