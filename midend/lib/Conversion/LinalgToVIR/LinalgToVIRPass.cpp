//===- LinalgToVIRPass.cpp - Linalg to VIR Dialect Conversion ---*- C++ -*-===//
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
// This file implements the conversion from linalg to VIR.
//
// The overall algorithm is similar to the vectorization in upstream MLIR.
// However, since VIR requires dynamic shape representations, memref
// transformations are performed first to align the shapes appropriately.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include "VIR/VIRDialect.h"
#include "VIR/VIROps.h"
#include "VIR/VIRTypes.h"

using namespace mlir;
using namespace buddy;

namespace {

/// Helper function to check leading static dims then return vector shape with
/// trailing dynamic dim (if any).
static LogicalResult
buildVIRVectorShape(linalg::LinalgOp op, SmallVectorImpl<int64_t> &shapeOut,
                    SmallVectorImpl<OpFoldResult> &ofrCommon, OpBuilder &b) {
  auto loc = op.getLoc();
  SmallVector<int64_t> staticLoopSizes = op.getStaticLoopRanges();
  SmallVector<OpFoldResult> commonShape;
  SmallVector<OpFoldResult> reducedShape;

  // XXX: Only the last loop dim can be dynamic for now (leading dynamic
  // dims not supported yet).
  for (int64_t i = 0, e = op.getNumLoops(); i < e; ++i) {
    int64_t sz = staticLoopSizes[i];
    auto itTy = op.getIteratorTypesArray()[i];
    if (ShapedType::isDynamic(sz)) {
      if (i != e - 1) {
        // Allow only the last dimension to be dynamic for now.
        return failure();
      }
      Value operand;
      unsigned operandDimPos = 0;
      if (failed(op.mapIterationSpaceDimToOperandDim(/*loopDimPos=*/i, operand,
                                                     operandDimPos))) {
        return failure();
      }
      Value dim = b.create<memref::DimOp>(loc, operand, operandDimPos);
      commonShape.push_back(dim);
    } else {
      commonShape.push_back(b.getIndexAttr(sz));
    }
    if (!linalg::isReductionIterator(itTy)) {
      reducedShape.push_back(commonShape.back());
    }
  }

  // Build VIR vector shape: copy loop sizes, but convert the last to dynamic
  // if it is dynamic, otherwise keep static.
  shapeOut.clear();
  shapeOut.reserve(staticLoopSizes.size());
  for (int64_t i = 0, e = op.getNumLoops(); i < e; ++i) {
    int64_t sz = staticLoopSizes[i];
    if (i == e - 1 && ShapedType::isDynamic(sz)) {
      shapeOut.push_back(ShapedType::kDynamic);
    } else {
      shapeOut.push_back(sz);
    }
  }

  ofrCommon = std::move(commonShape);
  return success();
}

/// Compute permutation to make indexing map reindexed. Adapted from
/// llvm/mlir/lib/Dialect/Linalg/Transforms/Vectorization.cpp, line 505-513.
static AffineMap reindexIndexingMap(AffineMap map) {
  assert(map.isProjectedPermutation(true) && "expected projected permutation");
  auto res = compressUnusedDims(map);
  assert(res.getNumDims() ==
             (res.getNumResults() - res.getNumOfZeroResults()) &&
         "expected reindexed map with same number of dims and results");
  return res;
}

/// Transform an input memref to align with iteration space order by
/// (a) expanding leading dims to match num loops,
/// (b) transposing per permutationMap, and
/// (c) subview to broadcast dims not present in indexing map (stride 0).
static Value transformInputMemrefForProjectedPermutation(
    OpBuilder &rewriter, Location loc, linalg::LinalgOp op,
    OpOperand *opOperand, AffineMap indexingMap,
    ArrayRef<OpFoldResult> commonShape) {
  auto memrefType = cast<MemRefType>(opOperand->get().getType());
  auto memrefShape = memrefType.getShape();

  // 1) Compute read and permutation maps of this operand.
  // e.g. (a, b, c) -> (c, b)
  // ==>  (d0, d1) -> (0, d1, d0)
  //        c   b      a   b   c
  auto readMap = inverseAndBroadcastProjectedPermutation(indexingMap);
  // Get the permuted dims. e.g. above, we want (d0, d1) -> (0, d1, d0)
  // the result is (0, d0, d1) with [0, 2, 1]
  SmallVector<unsigned> permutedDims;
  bool isPermBroadcast =
      readMap.isPermutationOfMinorIdentityWithBroadcasting(permutedDims);
  assert(isPermBroadcast && "expected permutation with broadcast");
  auto permutationMap =
      AffineMap::getPermutationMap(permutedDims, rewriter.getContext());

  // 2) Extend the memref shape to the rank of the common shape. The new
  // dimensions will be prepended at the front, and then permute with transpose.
  SmallVector<ReassociationIndices> reassociation;
  SmallVector<OpFoldResult> outputShape;
  SmallVector<int64_t> resultShape;
  // Build arguments for `memref.expand_shape`.
  auto numDimsToExpand = op.getNumLoops() - indexingMap.getNumResults();
  for (unsigned i = 0, e = numDimsToExpand; i < e; ++i) {
    outputShape.push_back(rewriter.getIndexAttr(1));
    resultShape.push_back(1);
  }
  for (unsigned i = 0, e = indexingMap.getNumResults(); i < e; ++i) {
    int64_t shapeVal = memrefShape[i];
    resultShape.push_back(shapeVal);
    if (ShapedType::isDynamic(shapeVal)) {
      Value dim = rewriter.create<memref::DimOp>(loc, opOperand->get(), i);
      outputShape.push_back(dim);
    } else {
      outputShape.push_back(rewriter.getIndexAttr(shapeVal));
    }
  }
  ReassociationIndices expandReassociation;
  for (unsigned i = 0, e = numDimsToExpand; i <= e; ++i) {
    expandReassociation.push_back(i);
  }
  reassociation.push_back(expandReassociation);
  for (unsigned i = 1, e = indexingMap.getNumResults(); i < e; ++i) {
    reassociation.push_back({static_cast<int64_t>(numDimsToExpand + i)});
  }

  Value expanded = rewriter.create<memref::ExpandShapeOp>(
      loc, resultShape, opOperand->get(), reassociation, outputShape);

  // 3) Transpose to align dims in iteration-space order.
  Value transposed = rewriter.create<memref::TransposeOp>(
      loc, expanded, AffineMapAttr::get(permutationMap));

  // 4) Subview to broadcast the dimensions that are not present.
  SmallVector<OpFoldResult> strides;
  SmallVector<OpFoldResult> offsets(op.getNumLoops(), rewriter.getIndexAttr(0));
  // Compute strides. All indexingMap inputs not present in the result are
  // 0, others are 1.
  for (unsigned i = 0, e = op.getNumLoops(); i < e; ++i) {
    auto dimExpr = rewriter.getAffineDimExpr(i);
    if (indexingMap.getResultPosition(dimExpr).has_value()) {
      strides.push_back(rewriter.getIndexAttr(1));
    } else {
      strides.push_back(rewriter.getIndexAttr(0));
    }
  }
  Value subview = rewriter.create<memref::SubViewOp>(loc, transposed, offsets,
                                                     commonShape, strides);
  return subview;
}

/// Transform an output memref to align with iteration space order (pure
/// transpose based on reindexed indexing map).
static Value
transformOutputMemrefForProjectedPermutation(OpBuilder &rewriter, Location loc,
                                             OpOperand *opOperand,
                                             AffineMap indexingMap) {
  // For output, just transpose it into the source dimension order.
  auto permutationMap = inversePermutation(reindexIndexingMap(indexingMap));
  auto transposed = rewriter.create<memref::TransposeOp>(
      loc, opOperand->get(), AffineMapAttr::get(permutationMap));
  return transposed;
}

/// Generic elementwise conversion: recreate the same op with VIR vector
/// operands and VIR vector result types. Fusion can be done in later passes.
/// Returns the created operation or nullptr on failure.
static Operation *createGenericElementwiseVIR(Operation *op,
                                              PatternRewriter &rewriter,
                                              ArrayRef<int64_t> virShape,
                                              DenseMap<Value, Value> &vm) {
  auto loc = op->getLoc();
  // Helper function to ensure that all operands are vectors.
  auto ensureVec = [&](Value v) -> Value {
    Value cur = v;
    if (auto it = vm.find(v); it != vm.end()) {
      cur = it->second;
    }
    if (cur && isa<buddy::vir::DynamicVectorType>(cur.getType())) {
      return cur;
    }
    // Broadcast scalar to vector type based on its scalar type.
    Type scalarTy = cur.getType();
    auto vecTy = buddy::vir::DynamicVectorType::get(virShape, scalarTy);
    return rewriter.create<buddy::vir::BroadcastOp>(loc, vecTy, cur);
  };

  // Collect VIR vector operands (broadcast scalars as needed).
  SmallVector<Value> vecOperands;
  vecOperands.reserve(op->getNumOperands());
  for (Value opd : op->getOperands()) {
    vecOperands.push_back(ensureVec(opd));
  }

  // arith.select requires scalar i1 condition in some MLIR versions. When the
  // condition is vectorized to !vir.vec<?xi1>, represent it explicitly as
  // vir.select and lower later.
  if (auto sel = dyn_cast<arith::SelectOp>(op)) {
    Type resElemTy = sel.getType();
    auto resTy = buddy::vir::DynamicVectorType::get(virShape, resElemTy);
    auto created = rewriter.create<buddy::vir::SelectOp>(
        loc, resTy, vecOperands[0], vecOperands[1], vecOperands[2]);
    return created.getOperation();
  }

  // Compute VIR vector result types matching original element types.
  SmallVector<Type> resultTypes;
  resultTypes.reserve(op->getNumResults());
  for (Type resTy : op->getResultTypes()) {
    // Expect element type results; wrap into VIR vector type with virShape.
    resultTypes.push_back(buddy::vir::DynamicVectorType::get(virShape, resTy));
  }

  // Recreate the op by identifier; fusion/combining is deferred to later.
  Operation *newOp = rewriter.create(loc, op->getName().getIdentifier(),
                                     vecOperands, resultTypes, op->getAttrs());
  return newOp;
}

static LogicalResult computeShapeAndVL(linalg::LinalgOp linalgOp,
                                       PatternRewriter &rewriter,
                                       SmallVectorImpl<int64_t> &virShape,
                                       SmallVectorImpl<OpFoldResult> &ofrCommon,
                                       Value &vlVal) {
  // TODO: If dynamic shapes are supported by VIR, we need to iterate trough the
  // ofrCommon to build the vector shape (and maybe no need to separate virShape
  // and vlVal).
  if (failed(buildVIRVectorShape(linalgOp, virShape, ofrCommon, rewriter))) {
    return rewriter.notifyMatchFailure(
        linalgOp, "leading dynamic dims not supported yet");
  }
  Location loc = linalgOp.getLoc();
  if (auto attr = dyn_cast<Attribute>(ofrCommon.back())) {
    vlVal = rewriter.create<arith::ConstantIndexOp>(
        loc, cast<IntegerAttr>(attr).getInt());
  } else if (auto v = dyn_cast<Value>(ofrCommon.back())) {
    vlVal = v;
  } else {
    return rewriter.notifyMatchFailure(linalgOp,
                                       "unexpected VL OpFoldResult kind");
  }
  return success();
}

static LogicalResult lowerReduceToScalarLoop(linalg::ReduceOp reduceOp,
                                             PatternRewriter &rewriter) {
  if (!reduceOp.hasPureBufferSemantics())
    return rewriter.notifyMatchFailure(reduceOp,
                                       "expected pure buffer semantics");
  if (reduceOp.getInputs().size() != 1 || reduceOp.getInits().size() != 1)
    return rewriter.notifyMatchFailure(reduceOp,
                                       "only single-input single-output reduce supported");

  Value input = *reduceOp.getInputs().begin();
  Value init = *reduceOp.getInits().begin();
  auto inTy = dyn_cast<MemRefType>(input.getType());
  auto outTy = dyn_cast<MemRefType>(init.getType());
  if (!inTy || !outTy)
    return rewriter.notifyMatchFailure(reduceOp, "expected memref operands");
  if (inTy.getRank() != 1 || outTy.getRank() != 0)
    return rewriter.notifyMatchFailure(
        reduceOp, "only rank-1 input to rank-0 output reduction supported");
  if (!inTy.getElementType().isF32() || !outTy.getElementType().isF32())
    return rewriter.notifyMatchFailure(reduceOp,
                                       "only f32 reduction supported");

  ArrayRef<int64_t> dims = reduceOp.getDimensions();
  if (dims.size() != 1 || dims[0] != 0)
    return rewriter.notifyMatchFailure(reduceOp,
                                       "only dimensions=[0] reduction supported");

  Block *body = reduceOp.getBody();
  if (!body || body->getNumArguments() != 2)
    return rewriter.notifyMatchFailure(reduceOp,
                                       "unexpected reduce combiner signature");
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return rewriter.notifyMatchFailure(reduceOp,
                                       "unexpected reduce combiner terminator");

  Operation *combiner = nullptr;
  for (Operation &inner : body->without_terminator()) {
    if (combiner)
      return rewriter.notifyMatchFailure(reduceOp,
                                         "only single-op combiner supported");
    combiner = &inner;
  }
  if (!combiner || yield.getOperand(0) != combiner->getResult(0))
    return rewriter.notifyMatchFailure(reduceOp,
                                       "unsupported reduce combiner shape");

  Location loc = reduceOp.getLoc();
  rewriter.setInsertionPoint(reduceOp);
  Value lower = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value upper = rewriter.create<memref::DimOp>(loc, input, 0);
  Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value initVal = rewriter.create<memref::LoadOp>(loc, init, ValueRange{});
  auto loop = rewriter.create<scf::ForOp>(loc, lower, upper, step,
                                          ValueRange{initVal});

  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();
    Value acc = loop.getRegionIterArgs().front();
    Value inVal = rewriter.create<memref::LoadOp>(loc, input, ValueRange{iv});

    auto mapCombinerOperand = [&](Value v) -> FailureOr<Value> {
      if (v == body->getArgument(0))
        return inVal;
      if (v == body->getArgument(1))
        return acc;
      return failure();
    };

    Value next;
    if (auto addf = dyn_cast<arith::AddFOp>(combiner)) {
      auto lhs = mapCombinerOperand(addf.getLhs());
      auto rhs = mapCombinerOperand(addf.getRhs());
      if (failed(lhs) || failed(rhs))
        return rewriter.notifyMatchFailure(
            reduceOp, "unsupported addf combiner operands");
      next = rewriter.create<arith::AddFOp>(loc, *lhs, *rhs);
    } else if (auto maxnumf = dyn_cast<arith::MaxNumFOp>(combiner)) {
      auto lhs = mapCombinerOperand(maxnumf.getLhs());
      auto rhs = mapCombinerOperand(maxnumf.getRhs());
      if (failed(lhs) || failed(rhs))
        return rewriter.notifyMatchFailure(
            reduceOp, "unsupported maxnumf combiner operands");
      next = rewriter.create<arith::MaxNumFOp>(loc, *lhs, *rhs);
    } else {
      return rewriter.notifyMatchFailure(
          reduceOp, "only addf/maxnumf combiner supported");
    }

    rewriter.create<scf::YieldOp>(loc, next);
  }

  Value reduced = loop.getResult(0);
  rewriter.create<memref::StoreOp>(loc, reduced, init, ValueRange{});
  rewriter.eraseOp(reduceOp);
  return success();
}

static LogicalResult lowerIndexOnlyGenericToScalarLoop(linalg::LinalgOp linalgOp,
                                                       PatternRewriter &rewriter) {
  auto genericOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation());
  if (!genericOp)
    return rewriter.notifyMatchFailure(linalgOp,
                                       "index-only fallback expects linalg.generic");
  if (genericOp.getNumLoops() != 1)
    return rewriter.notifyMatchFailure(genericOp,
                                       "only 1D index-only generic supported");
  auto itTy = genericOp.getIteratorTypesArray()[0];
  if (!linalg::isParallelIterator(itTy))
    return rewriter.notifyMatchFailure(genericOp,
                                       "only parallel iterator supported");
  if (genericOp.getNumDpsInputs() != 0 || genericOp.getNumDpsInits() != 1)
    return rewriter.notifyMatchFailure(genericOp,
                                       "only outs-only single-output generic supported");

  OpOperand *initOpd = genericOp.getDpsInitOperand(0);
  auto outTy = dyn_cast<MemRefType>(initOpd->get().getType());
  if (!outTy || outTy.getRank() != 1)
    return rewriter.notifyMatchFailure(genericOp,
                                       "output must be rank-1 memref");
  Type outElemTy = outTy.getElementType();
  if (!outElemTy.isIndex() && !isa<IntegerType>(outElemTy))
    return rewriter.notifyMatchFailure(
        genericOp, "output element type must be index or integer");

  Block &body = genericOp.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return rewriter.notifyMatchFailure(genericOp,
                                       "unexpected generic terminator");

  Value yielded = yield.getOperand(0);
  linalg::IndexOp idxOp = nullptr;
  arith::IndexCastOp castOp = nullptr;

  for (Operation &inner : body.without_terminator()) {
    if (auto idx = dyn_cast<linalg::IndexOp>(inner)) {
      if (idx.getDim() != 0)
        return rewriter.notifyMatchFailure(genericOp,
                                           "only linalg.index 0 supported");
      idxOp = idx;
      continue;
    }
    if (auto cast = dyn_cast<arith::IndexCastOp>(inner)) {
      castOp = cast;
      continue;
    }
    return rewriter.notifyMatchFailure(
        genericOp, (Twine("unsupported op in index-only generic fallback: ") +
                    inner.getName().getStringRef())
                       .str());
  }
  if (!idxOp)
    return rewriter.notifyMatchFailure(genericOp, "missing linalg.index");
  if (yielded != idxOp.getResult() &&
      (!castOp || yielded != castOp.getResult() || castOp.getIn() != idxOp)) {
    return rewriter.notifyMatchFailure(
        genericOp, "yield must be index or index_cast(index)");
  }

  Location loc = genericOp.getLoc();
  rewriter.setInsertionPoint(genericOp);
  Value out = initOpd->get();
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value n = rewriter.create<memref::DimOp>(loc, out, 0);
  auto loop = rewriter.create<scf::ForOp>(loc, c0, n, c1);
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();
    Value storeVal = iv;
    if (!outElemTy.isIndex()) {
      storeVal = rewriter.create<arith::IndexCastOp>(loc, outElemTy, iv);
    }
    rewriter.create<memref::StoreOp>(loc, storeVal, out, ValueRange{iv});
  }
  rewriter.eraseOp(genericOp);
  return success();
}

static bool isScalarCastOp(Operation &op) {
  return isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::ExtFOp,
             arith::TruncFOp, arith::SIToFPOp, arith::UIToFPOp,
             arith::FPToSIOp, arith::FPToUIOp, arith::BitcastOp,
             arith::IndexCastOp, arith::IndexCastUIOp>(op);
}

static LogicalResult
lowerSimpleCastGenericToScalarLoop(linalg::LinalgOp linalgOp,
                                   PatternRewriter &rewriter) {
  auto genericOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation());
  if (!genericOp)
    return rewriter.notifyMatchFailure(linalgOp,
                                       "cast fallback expects linalg.generic");
  if (genericOp.getNumLoops() != 1)
    return rewriter.notifyMatchFailure(genericOp, "only 1D generic supported");
  auto itTy = genericOp.getIteratorTypesArray()[0];
  if (!linalg::isParallelIterator(itTy))
    return rewriter.notifyMatchFailure(genericOp,
                                       "only parallel iterator supported");
  if (genericOp.getNumDpsInits() != 1)
    return rewriter.notifyMatchFailure(genericOp,
                                       "only single-output generic supported");

  auto is1DIdentityMap = [](AffineMap map) {
    return map.getNumDims() == 1 && map.getNumSymbols() == 0 &&
           map.getNumResults() == 1 && map.getResult(0).isFunctionOfDim(0);
  };
  for (AffineMap map : genericOp.getIndexingMapsArray()) {
    if (!is1DIdentityMap(map))
      return failure();
  }

  OpOperand *outOpd = genericOp.getDpsInitOperand(0);
  auto outTy = dyn_cast<MemRefType>(outOpd->get().getType());
  if (!outTy || outTy.getRank() != 1)
    return rewriter.notifyMatchFailure(genericOp,
                                       "output must be rank-1 memref");

  Block &body = genericOp.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return rewriter.notifyMatchFailure(genericOp,
                                       "unexpected generic terminator");
  Operation *castOp = nullptr;
  for (Operation &inner : body.without_terminator()) {
    if (castOp || !isScalarCastOp(inner))
      return failure();
    castOp = &inner;
  }
  if (!castOp || yield.getOperand(0) != castOp->getResult(0))
    return failure();

  Location loc = genericOp.getLoc();
  rewriter.setInsertionPoint(genericOp);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value upper = rewriter.create<memref::DimOp>(loc, outOpd->get(), 0);
  auto loop = rewriter.create<scf::ForOp>(loc, c0, upper, c1);
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();
    DenseMap<Value, Value> bbArgMap;
    for (OpOperand *opOperand : genericOp.getOpOperandsMatchingBBargs()) {
      Value bbArg = genericOp.getMatchingBlockArgument(opOperand);
      Value mapped;
      if (genericOp.isScalar(opOperand)) {
        mapped = opOperand->get();
      } else {
        mapped = rewriter.create<memref::LoadOp>(loc, opOperand->get(), iv);
      }
      bbArgMap[bbArg] = mapped;
    }

    SmallVector<Value> newOperands;
    newOperands.reserve(castOp->getNumOperands());
    for (Value operand : castOp->getOperands()) {
      if (!bbArgMap.contains(operand))
        return rewriter.notifyMatchFailure(genericOp,
                                           "unsupported operand in cast fallback");
      newOperands.push_back(bbArgMap.lookup(operand));
    }
    OperationState state(loc, castOp->getName().getStringRef());
    state.addOperands(newOperands);
    state.addTypes(castOp->getResultTypes());
    state.addAttributes(castOp->getAttrs());
    Operation *newCast = rewriter.create(state);
    rewriter.create<memref::StoreOp>(loc, newCast->getResult(0), outOpd->get(),
                                     iv);
  }
  rewriter.eraseOp(genericOp);
  return success();
}

static bool isSupportedGenericBodyOp(Operation &inner) {
  return isa<arith::ConstantOp, arith::AddFOp, arith::AddIOp, arith::SubFOp,
             arith::SubIOp, arith::MulFOp, arith::MulIOp, arith::DivFOp,
             arith::DivSIOp, arith::DivUIOp, arith::RemSIOp, arith::RemUIOp,
             arith::NegFOp, arith::CmpFOp, arith::CmpIOp, arith::SelectOp,
             arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::ShLIOp,
             arith::ShRUIOp, arith::ShRSIOp, arith::MaximumFOp,
             arith::MinimumFOp, arith::MaxSIOp, arith::MinSIOp,
             arith::MaxUIOp, arith::MinUIOp, arith::MaxNumFOp,
             math::ExpOp>(inner);
}

static buddy::vir::SetVLOp createSetVLRegion(PatternRewriter &rewriter,
                                             Location loc, Value vlVal) {
  auto setVl = rewriter.create<buddy::vir::SetVLOp>(
      loc, /*results=*/TypeRange{}, /*operands=*/ValueRange{vlVal});
  Region &region = setVl.getRegion();
  Block &block = region.emplaceBlock();
  rewriter.setInsertionPointToStart(&block);
  return setVl;
}

static LogicalResult lowerMatmulToVIR(linalg::MatmulOp matmulOp,
                                     PatternRewriter &rewriter) {
  if (!matmulOp.hasPureBufferSemantics())
    return rewriter.notifyMatchFailure(matmulOp,
                                       "expected pure buffer semantics");

  Location loc = matmulOp.getLoc();
  Value a = matmulOp.getInputs()[0];
  Value b = matmulOp.getInputs()[1];
  Value c = matmulOp.getOutputs()[0];

  auto aTy = dyn_cast<MemRefType>(a.getType());
  auto bTy = dyn_cast<MemRefType>(b.getType());
  auto cTy = dyn_cast<MemRefType>(c.getType());
  if (!aTy || !bTy || !cTy)
    return rewriter.notifyMatchFailure(matmulOp, "expected memref operands");
  if (aTy.getRank() != 2 || bTy.getRank() != 2 || cTy.getRank() != 2)
    return rewriter.notifyMatchFailure(matmulOp, "expected rank-2 memrefs");

  Type elemTy = aTy.getElementType();
  if (elemTy != bTy.getElementType() || elemTy != cTy.getElementType())
    return rewriter.notifyMatchFailure(matmulOp, "element types must match");
  if (!isa<FloatType>(elemTy))
    return rewriter.notifyMatchFailure(matmulOp,
                                       "only floating-point matmul supported");

  // Vectorize along N (the last dimension of B/C).
  Value n = rewriter.create<memref::DimOp>(loc, c, 1);

  // Create vir.set_vl region to host vector code.
  buddy::vir::SetVLOp setVl = createSetVLRegion(rewriter, loc, n);

  // Constants used inside the region.
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

  // Determine M/K. Prefer static when available, otherwise take dim from memref.
  Value mVal;
  if (!ShapedType::isDynamic(cTy.getShape()[0])) {
    mVal =
        rewriter.create<arith::ConstantIndexOp>(loc, cTy.getShape()[0]);
  } else {
    mVal = rewriter.create<memref::DimOp>(loc, c, 0);
  }

  Value kVal;
  if (!ShapedType::isDynamic(aTy.getShape()[1])) {
    kVal =
        rewriter.create<arith::ConstantIndexOp>(loc, aTy.getShape()[1]);
  } else {
    kVal = rewriter.create<memref::DimOp>(loc, a, 1);
  }

  auto vecTy = buddy::vir::DynamicVectorType::get({ShapedType::kDynamic}, elemTy);

  auto loopM = rewriter.create<affine::AffineForOp>(
      loc, ValueRange{c0}, rewriter.getDimIdentityMap(), ValueRange{mVal},
      rewriter.getDimIdentityMap(), /*step=*/1,
      /*iterArgs=*/std::nullopt,
      [&](OpBuilder &bld, Location bodyLoc, Value i, ValueRange) {
        OpBuilder &builder = bld;
        // acc = load(C[i, 0:]) as a vector along N.
        Value acc = builder
                        .create<buddy::vir::LoadOp>(bodyLoc, vecTy, c,
                                                    ValueRange{i, c0})
                        .getResult();

        auto loopK = builder.create<affine::AffineForOp>(
            bodyLoc, ValueRange{c0}, rewriter.getDimIdentityMap(),
            ValueRange{kVal}, rewriter.getDimIdentityMap(), /*step=*/1,
            /*iterArgs=*/ValueRange{acc},
            [&](OpBuilder &kb, Location kLoc, Value k, ValueRange iterArgs) {
              OpBuilder &builderK = kb;
              Value accIn = iterArgs[0];
              // aScalar = A[i, k]
              Value aScalar =
                  builderK.create<memref::LoadOp>(kLoc, a, ValueRange{i, k});
              // aVec = broadcast(aScalar)
              Value aVec =
                  builderK
                      .create<buddy::vir::BroadcastOp>(kLoc, vecTy, aScalar)
                      .getResult();
              // bVec = load(B[k, 0:]) as a vector along N.
              Value bVec =
                  builderK
                      .create<buddy::vir::LoadOp>(kLoc, vecTy, b,
                                                  ValueRange{k, c0})
                      .getResult();
              // accOut = fma(aVec, bVec, accIn)
              Value accOut =
                  builderK
                      .create<buddy::vir::FMAOp>(kLoc, vecTy, aVec, bVec, accIn)
                      .getResult();
              builderK.create<affine::AffineYieldOp>(kLoc, accOut);
            });
        Value finalAcc = loopK.getResult(0);

        // store acc back to C[i, 0:].
        builder.create<buddy::vir::StoreOp>(bodyLoc, finalAcc, c,
                                            ValueRange{i, c0});
        builder.create<affine::AffineYieldOp>(bodyLoc);
      });
  (void)loopM;

  // Close the set_vl region.
  rewriter.create<vector::YieldOp>(loc);

  rewriter.replaceOp(matmulOp, setVl.getResults());
  return success();
}

/// Expand & transpose source memrefs to the common shape, and map them into
/// `transformedMemRefs`.
static LogicalResult transformProjectedPermutationOperands(
    linalg::LinalgOp linalgOp, PatternRewriter &rewriter,
    ArrayRef<OpFoldResult> commonShape,
    DenseMap<Value, Value> &transformedMemRefs) {
  Location loc = linalgOp.getLoc();
  for (OpOperand *opOperand : linalgOp.getOpOperandsMatchingBBargs()) {
    if (linalgOp.isScalar(opOperand)) {
      continue;
    }
    auto indexingMap = linalgOp.getMatchingIndexingMap(opOperand);
    if (!indexingMap.isProjectedPermutation()) {
      return rewriter.notifyMatchFailure(
          linalgOp, "non-projected permutation operands not supported yet");
    }
    if (linalgOp.isDpsInput(opOperand)) {
      Value sub = transformInputMemrefForProjectedPermutation(
          rewriter, loc, linalgOp, opOperand, indexingMap, commonShape);
      transformedMemRefs[opOperand->get()] = sub;
    } else {
      Value tr = transformOutputMemrefForProjectedPermutation(
          rewriter, loc, opOperand, indexingMap);
      transformedMemRefs[opOperand->get()] = tr;
    }
  }
  return success();
}

/// Load input memrefs into VIR vectors.
static LogicalResult mapInputsToVIRVectors(linalg::LinalgOp linalgOp,
                                           PatternRewriter &rewriter,
                                           ArrayRef<int64_t> virShape,
                                           const DenseMap<Value, Value> &memMap,
                                           IRMapping &valueMap) {
  Location loc = linalgOp.getLoc();
  for (OpOperand *opOperand : linalgOp.getOpOperandsMatchingBBargs()) {
    auto bbArg = linalgOp.getMatchingBlockArgument(opOperand);
    if (linalgOp.isScalar(opOperand)) {
      valueMap.map(bbArg, opOperand->get());
      continue;
    }
    auto indexingMap = linalgOp.getMatchingIndexingMap(opOperand);
    if (!indexingMap.isProjectedPermutation()) {
      // TODO: Support non-projected-permutation with gather.
      return rewriter.notifyMatchFailure(
          linalgOp, "non-projected permutation operands not supported yet");
    }
    Value base = memMap.lookup(opOperand->get());
    // Load the whole memref as a conceptual vector.
    SmallVector<Value> zeroIdx(0);
    auto memrefTy = cast<MemRefType>(opOperand->get().getType());
    auto vecTy =
        buddy::vir::DynamicVectorType::get(virShape, memrefTy.getElementType());
    auto loaded =
        rewriter.create<buddy::vir::LoadOp>(loc, vecTy, base, zeroIdx);
    valueMap.map(bbArg, loaded.getResult());
  }
  return success();
}

/// Convert the body of a LinalgOp to use VIR operations. Since all inputs are
/// aligned to a common shape, most operations can be transformed directly.
static LogicalResult convertBodyToVIR(linalg::LinalgOp linalgOp,
                                      PatternRewriter &rewriter,
                                      ArrayRef<int64_t> virShape,
                                      IRMapping &valueMap,
                                      DenseMap<Value, Value> &vm) {
  Location loc = linalgOp.getLoc();
  for (auto it : valueMap.getValueMap())
    vm[it.first] = it.second;
  for (Operation &inner : linalgOp.getBlock()->getOperations()) {
    if (isa<linalg::YieldOp>(inner)) {
      // Handled by storeYieldValues
      // TODO: For 1-D reduction, this should map to a return.reduce op.
      continue;
    }
    if (auto cst = dyn_cast<arith::ConstantOp>(inner)) {
      // Broadcast constant scalar to a vector matching its scalar type.
      auto vecTy = buddy::vir::DynamicVectorType::get(virShape, cst.getType());
      auto v =
          rewriter.create<buddy::vir::BroadcastOp>(loc, vecTy, cst.getResult());
      vm[cst.getResult()] = v.getResult();
      continue;
    }
    if (isa<linalg::IndexOp>(inner)) {
      // TODO: linalg.index basically extracts the underlying loop induction
      // variables. This should be lowered with vid (RVV) or stepvector (LLVM),
      // or just broadcasting normal induction variables. This requires both the
      // higher construction of VIR and backend supports.
      return rewriter.notifyMatchFailure(linalgOp,
                                         "linalg.index not supported");
    }
    if (OpTrait::hasElementwiseMappableTraits(&inner)) {
      // General conversion
      if (Operation *nw =
              createGenericElementwiseVIR(&inner, rewriter, virShape, vm)) {
        for (auto [oldRes, newRes] :
             llvm::zip(inner.getResults(), nw->getResults())) {
          vm[oldRes] = newRes;
        }
        continue;
      }
    }
    // TODO: Support more operations, including supporting reduction with
    // `matchReduction` in MLIR.
    return rewriter.notifyMatchFailure(
        linalgOp,
        (Twine("unsupported inner op: ") + inner.getName().getStringRef())
            .str());
  }
  return success();
}

static LogicalResult storeYieldValues(linalg::LinalgOp linalgOp,
                                      PatternRewriter &rewriter,
                                      const DenseMap<Value, Value> &memMap,
                                      IRMapping &valueMap,
                                      DenseMap<Value, Value> &vm) {
  Location loc = linalgOp.getLoc();
  auto yield = cast<linalg::YieldOp>(linalgOp.getBlock()->getTerminator());
  for (auto [i, yv] : llvm::enumerate(yield.getOperands())) {
    Value mapped = vm.lookup(yv);
    auto initOpd = linalgOp.getDpsInitOperand(i);
    auto outElemTy =
        cast<MemRefType>(initOpd->get().getType()).getElementType();

    // Derive shape again to form the target vector type.
    SmallVector<int64_t> virShape;
    SmallVector<OpFoldResult> tmp;
    if (failed(buildVIRVectorShape(linalgOp, virShape, tmp, rewriter))) {
      return rewriter.notifyMatchFailure(linalgOp, "invalid vector shape");
    }
    auto outVecTy = buddy::vir::DynamicVectorType::get(virShape, outElemTy);

    // If the yield value is scalar (e.g., linalg.fill), broadcast it to the
    // output vector type before storing.
    if (mapped) {
      if (!isa<buddy::vir::DynamicVectorType>(mapped.getType())) {
        mapped =
            rewriter.create<buddy::vir::BroadcastOp>(loc, outVecTy, mapped);
      }
    } else if (valueMap.contains(yv)) {
      Value scalar = valueMap.lookup(yv);
      mapped = rewriter.create<buddy::vir::BroadcastOp>(loc, outVecTy, scalar);
    }
    auto indexingMap = linalgOp.getMatchingIndexingMap(initOpd);
    if (!indexingMap.isProjectedPermutation()) {
      // TODO: Support non-projected-permutation with scatter.
      return rewriter.notifyMatchFailure(
          linalgOp, "non-projected permutation outputs not supported yet");
    }
    Value base = memMap.lookup(initOpd->get());
    // Use implicit indices for vir.store. The lowering to vector will interpret
    // empty indices as using the leading-dim IVs (if any) and the VL IV.
    SmallVector<Value> emptyIdx;
    rewriter.create<buddy::vir::StoreOp>(loc, mapped, base, emptyIdx);
  }
  return success();
}

struct LinalgGenericToVIRPattern : public RewritePattern {
  LinalgGenericToVIRPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return rewriter.notifyMatchFailure(op, "expected linalg op");
    }
    if (!linalgOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(op, "expected pure buffer semantics");
    }
    // Guardrail: current VIR lowering handles elementwise/matmul-like patterns.
    // Reject reduction/index-based forms early to avoid partial rewrites.
    for (auto itTy : linalgOp.getIteratorTypesArray()) {
      if (linalg::isReductionIterator(itTy)) {
        return rewriter.notifyMatchFailure(
            linalgOp, "reduction iterators not supported in generic VIR lowering");
      }
    }
    for (unsigned i = 0, e = linalgOp.getNumDpsInits(); i < e; ++i) {
      OpOperand *init = linalgOp.getDpsInitOperand(i);
      auto ty = dyn_cast<MemRefType>(init->get().getType());
      if (ty && ty.getRank() == 0) {
        return rewriter.notifyMatchFailure(
            linalgOp, "rank-0 output memref not supported in generic VIR lowering");
      }
    }
    bool hasIndexSemantics = false;
    bool hasCastSemantics = false;
    for (Operation &inner : linalgOp.getBlock()->getOperations()) {
      if (isa<linalg::YieldOp>(inner))
        continue;
      if (isa<linalg::IndexOp>(inner)) {
        hasIndexSemantics = true;
        continue;
      }
      if (isScalarCastOp(inner)) {
        hasCastSemantics = true;
        continue;
      }
      if (!isSupportedGenericBodyOp(inner)) {
        return rewriter.notifyMatchFailure(
            linalgOp, (Twine("unsupported generic inner op: ") +
                       inner.getName().getStringRef())
                          .str());
      }
    }
    if (hasIndexSemantics) {
      if (succeeded(lowerIndexOnlyGenericToScalarLoop(linalgOp, rewriter)))
        return success();
      return rewriter.notifyMatchFailure(
          linalgOp, "linalg.index generic shape not supported yet");
    }
    if (hasCastSemantics) {
      if (succeeded(lowerSimpleCastGenericToScalarLoop(linalgOp, rewriter)))
        return success();
      return rewriter.notifyMatchFailure(
          linalgOp, "cast generic shape not supported yet");
    }
    // Compute VIR vector shape and VL value.
    SmallVector<int64_t> virShape;
    SmallVector<OpFoldResult> commonShape;
    Value vlVal;
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(linalgOp);
      if (failed(computeShapeAndVL(linalgOp, rewriter, virShape, commonShape,
                                   vlVal)))
        return failure();
    }

    Location loc = linalgOp.getLoc();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(linalgOp);

    // Create vir.set_vl region to host vector code.
    (void)createSetVLRegion(rewriter, loc, vlVal);

    // Prepare transformed memrefs and mapping from bbargs to vir vectors.
    DenseMap<Value, Value> transformedMemRefs;
    IRMapping valueMap;

    // Carry values defined above.
    llvm::SetVector<Value> valuesDefinedAbove;
    mlir::getUsedValuesDefinedAbove(linalgOp->getRegion(0), valuesDefinedAbove);
    valueMap.map(valuesDefinedAbove.getArrayRef(),
                 valuesDefinedAbove.getArrayRef());

    // 1) Transform inputs and outputs with projected permutations.
    if (failed(transformProjectedPermutationOperands(
            linalgOp, rewriter, commonShape, transformedMemRefs)))
      return failure();

    // 2) Load inputs into VIR vectors (projected permutation only).
    if (failed(mapInputsToVIRVectors(linalgOp, rewriter, virShape,
                                     transformedMemRefs, valueMap)))
      return failure();

    // 3) Convert the body to VIR elementwise ops.
    DenseMap<Value, Value> vm;
    if (failed(convertBodyToVIR(linalgOp, rewriter, virShape, valueMap, vm)))
      return failure();

    // 4) TODO: Handle reductions when reduction is supported in VIR.

    // 5) Store yielded values into outputs.
    if (failed(storeYieldValues(linalgOp, rewriter, transformedMemRefs,
                                valueMap, vm))) {
      return failure();
    }

    // Close the set_vl region by ending the block (no explicit terminator).
    rewriter.create<vector::YieldOp>(loc);

    // Erase original op.
    rewriter.eraseOp(linalgOp);
    return success();
  }
};

struct LinalgReduceToVIRPattern : public RewritePattern {
  LinalgReduceToVIRPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/3, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto reduceOp = dyn_cast<linalg::ReduceOp>(op);
    if (!reduceOp)
      return rewriter.notifyMatchFailure(op, "expected linalg.reduce");
    return lowerReduceToScalarLoop(reduceOp, rewriter);
  }
};

struct LinalgMatmulToVIRPattern : public RewritePattern {
  LinalgMatmulToVIRPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto matmulOp = dyn_cast<linalg::MatmulOp>(op);
    if (!matmulOp)
      return rewriter.notifyMatchFailure(op, "expected linalg.matmul");
    return lowerMatmulToVIR(matmulOp, rewriter);
  }
};

class LinalgToVIRPass
    : public PassWrapper<LinalgToVIRPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToVIRPass)
  LinalgToVIRPass() = default;
  LinalgToVIRPass(const LinalgToVIRPass &) {}

  StringRef getArgument() const final { return "lower-linalg-to-vir"; }
  StringRef getDescription() const final {
    return "Lower Linalg Dialect to VIR Dialect (dynamic vectors).";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LinalgReduceToVIRPattern>(ctx);
    patterns.add<LinalgMatmulToVIRPattern>(ctx);
    patterns.add<LinalgGenericToVIRPattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, memref::MemRefDialect,
                    scf::SCFDialect, buddy::vir::VIRDialect,
                    vector::VectorDialect>();
  }
};

} // namespace

namespace mlir {
namespace buddy {

void registerLinalgToVIRPass() { PassRegistration<LinalgToVIRPass>(); }

} // namespace buddy
} // namespace mlir
