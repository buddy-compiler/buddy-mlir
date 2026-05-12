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

/// Helper function to check leading static dims, build a VIR vector type shape
/// whose last dimension is always dynamic, and preserve the concrete logical
/// iteration extents in `ofrCommon` for downstream VL derivation.
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

  // Keep leading dimensions unchanged, but always model the vectorized last
  // dimension as dynamic in `shapeOut`. The concrete last extent is preserved
  // separately in `commonShape`/`ofrCommon`, so callers must use
  // `ofrCommon.back()` rather than `shapeOut.back()` when deriving VL.
  shapeOut.clear();
  shapeOut.reserve(staticLoopSizes.size());
  for (int64_t i = 0, e = op.getNumLoops(); i < e; ++i) {
    if (i == e - 1) {
      shapeOut.push_back(ShapedType::kDynamic);
    } else {
      shapeOut.push_back(staticLoopSizes[i]);
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

  // arith.extf/truncf cannot directly produce !vir.vec types. Represent them as
  // dedicated VIR ops and lower later in VIRToVector.
  if (auto extf = dyn_cast<arith::ExtFOp>(op)) {
    auto outTy = buddy::vir::DynamicVectorType::get(virShape, extf.getType());
    auto created =
        rewriter.create<buddy::vir::ExtFOp>(loc, outTy, vecOperands[0]);
    return created.getOperation();
  }
  if (auto truncf = dyn_cast<arith::TruncFOp>(op)) {
    auto outTy = buddy::vir::DynamicVectorType::get(virShape, truncf.getType());
    auto created =
        rewriter.create<buddy::vir::TruncFOp>(loc, outTy, vecOperands[0]);
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

static buddy::vir::SetVLOp createSetVLRegion(PatternRewriter &rewriter,
                                             Location loc, Value vlVal);

enum class SupportedReduceCombinerKind { AddF, MaxNumF, MaxSI };

static FailureOr<SupportedReduceCombinerKind>
getSupportedReduceCombinerKind(linalg::ReduceOp reduceOp,
                               PatternRewriter &rewriter) {
  Block *body = reduceOp.getBody();
  if (!body || body->getNumArguments() != 2)
    return failure();

  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return failure();

  Operation *combiner = nullptr;
  for (Operation &inner : body->without_terminator()) {
    if (combiner)
      return failure();
    combiner = &inner;
  }
  if (!combiner || yield.getOperand(0) != combiner->getResult(0))
    return failure();

  if (isa<arith::AddFOp>(combiner))
    return SupportedReduceCombinerKind::AddF;
  if (isa<arith::MaxNumFOp>(combiner))
    return SupportedReduceCombinerKind::MaxNumF;
  if (isa<arith::MaxSIOp>(combiner))
    return SupportedReduceCombinerKind::MaxSI;
  return failure();
}

static LogicalResult lowerReduceToScalarLoop(linalg::ReduceOp reduceOp,
                                             PatternRewriter &rewriter) {
  if (!reduceOp.hasPureBufferSemantics())
    return rewriter.notifyMatchFailure(reduceOp,
                                       "expected pure buffer semantics");
  if (reduceOp.getInputs().size() != 1 || reduceOp.getInits().size() != 1)
    return rewriter.notifyMatchFailure(
        reduceOp, "only single-input single-output reduce supported");

  Value input = *reduceOp.getInputs().begin();
  Value init = *reduceOp.getInits().begin();
  auto inTy = dyn_cast<MemRefType>(input.getType());
  auto outTy = dyn_cast<MemRefType>(init.getType());
  if (!inTy || !outTy)
    return rewriter.notifyMatchFailure(reduceOp, "expected memref operands");
  Type inElemTy = inTy.getElementType();
  Type outElemTy = outTy.getElementType();
  if (inElemTy != outElemTy)
    return rewriter.notifyMatchFailure(reduceOp,
                                       "input/output element types must match");

  auto combinerKind = getSupportedReduceCombinerKind(reduceOp, rewriter);
  if (failed(combinerKind))
    return rewriter.notifyMatchFailure(
        reduceOp,
        "unsupported reduce combiner (expected single-op addf/maxnumf/maxsi "
        "with linalg.yield)");

  switch (*combinerKind) {
  case SupportedReduceCombinerKind::AddF:
  case SupportedReduceCombinerKind::MaxNumF:
    if (!isa<FloatType>(inElemTy))
      return rewriter.notifyMatchFailure(
          reduceOp,
          "floating-point reduce combiner requires float element type");
    break;
  case SupportedReduceCombinerKind::MaxSI:
    if (!isa<IntegerType>(inElemTy))
      return rewriter.notifyMatchFailure(
          reduceOp, "arith.maxsi reduction requires integer element type");
    break;
  }

  ArrayRef<int64_t> dims = reduceOp.getDimensions();

  Location loc = reduceOp.getLoc();
  rewriter.setInsertionPoint(reduceOp);
  Value lower = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  auto kindToString = [&](SupportedReduceCombinerKind k) -> StringRef {
    switch (k) {
    case SupportedReduceCombinerKind::AddF:
      return "add";
    case SupportedReduceCombinerKind::MaxNumF:
      return "maxnum";
    case SupportedReduceCombinerKind::MaxSI:
      return "maxsi";
    }
    llvm_unreachable("unknown reduce combiner");
  };

  // Case A: rank-1 -> rank-0, dimensions = [0].
  // Reduce the full 1D buffer into a scalar accumulator.
  if (inTy.getRank() == 1 && outTy.getRank() == 0) {
    if (dims.size() != 1 || dims[0] != 0)
      return rewriter.notifyMatchFailure(
          reduceOp, "rank-1 -> rank-0 reduction requires dimensions=[0]");

    Value n = rewriter.create<memref::DimOp>(loc, input, 0);
    Type elemTy = inTy.getElementType();
    auto accBufTy = MemRefType::get({}, elemTy);
    Value accBuf = rewriter.create<memref::AllocaOp>(loc, accBufTy);
    Value initVal = rewriter.create<memref::LoadOp>(loc, init, ValueRange{});
    rewriter.create<memref::StoreOp>(loc, initVal, accBuf, ValueRange{});

    auto setVl = createSetVLRegion(rewriter, loc, n);
    auto vecTy =
        buddy::vir::DynamicVectorType::get({ShapedType::kDynamic}, elemTy);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value loaded =
        rewriter.create<buddy::vir::LoadOp>(loc, vecTy, input, ValueRange{c0})
            .getResult();
    Value acc = rewriter.create<memref::LoadOp>(loc, accBuf, ValueRange{});
    Value reduced = rewriter
                        .create<buddy::vir::ReduceOp>(
                            loc, elemTy, loaded, acc,
                            rewriter.getStringAttr(kindToString(*combinerKind)))
                        .getResult();
    rewriter.create<memref::StoreOp>(loc, reduced, accBuf, ValueRange{});
    rewriter.create<vector::YieldOp>(loc);

    rewriter.setInsertionPointAfter(setVl);
    Value finalVal = rewriter.create<memref::LoadOp>(loc, accBuf, ValueRange{});
    rewriter.create<memref::StoreOp>(loc, finalVal, init, ValueRange{});
    rewriter.eraseOp(reduceOp);
    return success();
  }

  // Case B: rank-2 -> rank-1, dimensions = [1].
  // Reduce each row of the MxN input into one output element.
  if (inTy.getRank() == 2 && outTy.getRank() == 1) {
    if (dims.size() != 1)
      return rewriter.notifyMatchFailure(
          reduceOp, "rank-2 -> rank-1 reduction requires one dimension");

    int64_t inStaticM = inTy.getShape()[0];
    int64_t inStaticN = inTy.getShape()[1];
    int64_t outStaticM = outTy.getShape()[0];
    Type elemTy = inTy.getElementType();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    if (dims[0] == 1) {
      if (!ShapedType::isDynamic(inStaticM) &&
          !ShapedType::isDynamic(outStaticM) && inStaticM != outStaticM) {
        return rewriter.notifyMatchFailure(
            reduceOp, "input/output leading dimension mismatch");
      }

      Value upperM = rewriter.create<memref::DimOp>(loc, input, 0);
      Value upperN = rewriter.create<memref::DimOp>(loc, input, 1);
      auto outerLoop = rewriter.create<scf::ForOp>(loc, lower, upperM, step);

      {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointToStart(outerLoop.getBody());
        Value i = outerLoop.getInductionVar();
        auto accBufTy = MemRefType::get({}, elemTy);
        Value accBuf = rewriter.create<memref::AllocaOp>(loc, accBufTy);
        Value initVal =
            rewriter.create<memref::LoadOp>(loc, init, ValueRange{i});
        rewriter.create<memref::StoreOp>(loc, initVal, accBuf, ValueRange{});

        auto setVl = rewriter.create<buddy::vir::SetVLOp>(
            loc, /*results=*/TypeRange{}, /*operands=*/ValueRange{upperN});
        Region &region = setVl.getRegion();
        Block &block = region.emplaceBlock();
        rewriter.setInsertionPointToStart(&block);

        auto vecTy =
            buddy::vir::DynamicVectorType::get({ShapedType::kDynamic}, elemTy);
        Value row = rewriter
                        .create<buddy::vir::LoadOp>(loc, vecTy, input,
                                                    ValueRange{i, c0})
                        .getResult();
        Value acc = rewriter.create<memref::LoadOp>(loc, accBuf, ValueRange{});
        Value reduced =
            rewriter
                .create<buddy::vir::ReduceOp>(
                    loc, elemTy, row, acc,
                    rewriter.getStringAttr(kindToString(*combinerKind)))
                .getResult();
        rewriter.create<memref::StoreOp>(loc, reduced, accBuf, ValueRange{});
        rewriter.create<vector::YieldOp>(loc);

        rewriter.setInsertionPointAfter(setVl);
        Value finalVal =
            rewriter.create<memref::LoadOp>(loc, accBuf, ValueRange{});
        rewriter.create<memref::StoreOp>(loc, finalVal, init, ValueRange{i});
      }

      rewriter.eraseOp(reduceOp);
      return success();
    }

    // Case C: rank-2 -> rank-1, dimensions = [0].
    // First transpose MxN -> NxM so reducing the original leading dimension
    // becomes a reduction across one transposed row. The transpose view has a
    // non-unit stride on the minor dimension, so we materialize it into a
    // contiguous NxM scratch buffer before lowering further; otherwise
    // -lower-vir-to-vector would reject the eventual vector.load because the
    // innermost dimension would not be unit-stride.
    if (dims[0] == 0) {
      int64_t outStaticN = outTy.getShape()[0];
      if (ShapedType::isDynamic(inStaticM) ||
          ShapedType::isDynamic(inStaticN) ||
          ShapedType::isDynamic(outStaticN)) {
        return rewriter.notifyMatchFailure(
            reduceOp, "rank-2 -> rank-1 dimensions=[0] requires static M/N");
      }
      if (inStaticN != outStaticN)
        return rewriter.notifyMatchFailure(
            reduceOp, "input/output trailing dimension mismatch");

      Value upperN = rewriter.create<arith::ConstantIndexOp>(loc, inStaticN);
      Value upperM = rewriter.create<arith::ConstantIndexOp>(loc, inStaticM);
      SmallVector<int64_t> permutation = {1, 0};
      auto permutationMap =
          AffineMap::getPermutationMap(permutation, rewriter.getContext());
      Value transposed = rewriter.create<memref::TransposeOp>(
          loc, input, AffineMapAttr::get(permutationMap));
      auto scratchTy = MemRefType::get({inStaticN, inStaticM}, elemTy);
      Value scratch = rewriter.create<memref::AllocOp>(loc, scratchTy);
      rewriter.create<memref::CopyOp>(loc, transposed, scratch);
      auto outerLoop = rewriter.create<scf::ForOp>(loc, lower, upperN, step);

      {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointToStart(outerLoop.getBody());
        Value j = outerLoop.getInductionVar();
        auto accBufTy = MemRefType::get({}, elemTy);
        Value accBuf = rewriter.create<memref::AllocaOp>(loc, accBufTy);
        Value initVal =
            rewriter.create<memref::LoadOp>(loc, init, ValueRange{j});
        rewriter.create<memref::StoreOp>(loc, initVal, accBuf, ValueRange{});

        auto setVl = rewriter.create<buddy::vir::SetVLOp>(
            loc, /*results=*/TypeRange{}, /*operands=*/ValueRange{upperM});
        Region &region = setVl.getRegion();
        Block &block = region.emplaceBlock();
        rewriter.setInsertionPointToStart(&block);

        auto vecTy =
            buddy::vir::DynamicVectorType::get({ShapedType::kDynamic}, elemTy);
        Value col = rewriter
                        .create<buddy::vir::LoadOp>(loc, vecTy, scratch,
                                                    ValueRange{j, c0})
                        .getResult();
        Value acc = rewriter.create<memref::LoadOp>(loc, accBuf, ValueRange{});
        Value reduced =
            rewriter
                .create<buddy::vir::ReduceOp>(
                    loc, elemTy, col, acc,
                    rewriter.getStringAttr(kindToString(*combinerKind)))
                .getResult();
        rewriter.create<memref::StoreOp>(loc, reduced, accBuf, ValueRange{});
        rewriter.create<vector::YieldOp>(loc);

        rewriter.setInsertionPointAfter(setVl);
        Value finalVal =
            rewriter.create<memref::LoadOp>(loc, accBuf, ValueRange{});
        rewriter.create<memref::StoreOp>(loc, finalVal, init, ValueRange{j});
      }
      rewriter.create<memref::DeallocOp>(loc, scratch);

      rewriter.eraseOp(reduceOp);
      return success();
    }

    return rewriter.notifyMatchFailure(
        reduceOp, "rank-2 -> rank-1 reduction requires dimensions=[0] or [1]");
  }

  // Case D: rank-3 -> rank-2, dimensions = [0].
  // Transpose MxNxK -> NxKxM so reducing the original leading dimension becomes
  // a reduction across one contiguous length-M slice per (j, k). The transpose
  // view is still strided on the minor dimension, so we materialize it into a
  // contiguous NxKxM scratch buffer; otherwise -lower-vir-to-vector would
  // reject the eventual vector.load because the innermost dimension would not
  // be unit-stride.
  if (inTy.getRank() == 3 && outTy.getRank() == 2) {
    if (dims.size() != 1 || dims[0] != 0)
      return rewriter.notifyMatchFailure(
          reduceOp, "rank-3 -> rank-2 reduction requires dimensions=[0]");

    int64_t inStaticM = inTy.getShape()[0];
    int64_t inStaticN = inTy.getShape()[1];
    int64_t inStaticK = inTy.getShape()[2];
    int64_t outStaticN = outTy.getShape()[0];
    int64_t outStaticK = outTy.getShape()[1];
    if (ShapedType::isDynamic(inStaticM) || ShapedType::isDynamic(inStaticN) ||
        ShapedType::isDynamic(inStaticK) || ShapedType::isDynamic(outStaticN) ||
        ShapedType::isDynamic(outStaticK)) {
      return rewriter.notifyMatchFailure(
          reduceOp, "rank-3 -> rank-2 dimensions=[0] requires static M/N/K");
    }
    if (inStaticN != outStaticN || inStaticK != outStaticK)
      return rewriter.notifyMatchFailure(
          reduceOp, "input/output trailing dimensions mismatch");

    Type elemTy = inTy.getElementType();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value upperN = rewriter.create<arith::ConstantIndexOp>(loc, inStaticN);
    Value upperK = rewriter.create<arith::ConstantIndexOp>(loc, inStaticK);
    Value upperM = rewriter.create<arith::ConstantIndexOp>(loc, inStaticM);
    SmallVector<int64_t> permutation = {1, 2, 0};
    auto permutationMap =
        AffineMap::getPermutationMap(permutation, rewriter.getContext());

    Value transposed = rewriter.create<memref::TransposeOp>(
        loc, input, AffineMapAttr::get(permutationMap));
    auto scratchTy = MemRefType::get({inStaticN, inStaticK, inStaticM}, elemTy);
    Value scratch = rewriter.create<memref::AllocOp>(loc, scratchTy);
    rewriter.create<memref::CopyOp>(loc, transposed, scratch);
    auto outerLoop = rewriter.create<scf::ForOp>(loc, lower, upperN, step);

    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(outerLoop.getBody());
      Value j = outerLoop.getInductionVar();
      auto innerLoop = rewriter.create<scf::ForOp>(loc, lower, upperK, step);

      {
        OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        Value k = innerLoop.getInductionVar();
        auto accBufTy = MemRefType::get({}, elemTy);
        Value accBuf = rewriter.create<memref::AllocaOp>(loc, accBufTy);
        Value initVal =
            rewriter.create<memref::LoadOp>(loc, init, ValueRange{j, k});
        rewriter.create<memref::StoreOp>(loc, initVal, accBuf, ValueRange{});

        auto setVl = rewriter.create<buddy::vir::SetVLOp>(
            loc, /*results=*/TypeRange{}, /*operands=*/ValueRange{upperM});
        Region &region = setVl.getRegion();
        Block &block = region.emplaceBlock();
        rewriter.setInsertionPointToStart(&block);

        auto vecTy =
            buddy::vir::DynamicVectorType::get({ShapedType::kDynamic}, elemTy);
        Value slice = rewriter
                          .create<buddy::vir::LoadOp>(loc, vecTy, scratch,
                                                      ValueRange{j, k, c0})
                          .getResult();
        Value acc = rewriter.create<memref::LoadOp>(loc, accBuf, ValueRange{});
        Value reduced =
            rewriter
                .create<buddy::vir::ReduceOp>(
                    loc, elemTy, slice, acc,
                    rewriter.getStringAttr(kindToString(*combinerKind)))
                .getResult();
        rewriter.create<memref::StoreOp>(loc, reduced, accBuf, ValueRange{});
        rewriter.create<vector::YieldOp>(loc);

        rewriter.setInsertionPointAfter(setVl);
        Value finalVal =
            rewriter.create<memref::LoadOp>(loc, accBuf, ValueRange{});
        rewriter.create<memref::StoreOp>(loc, finalVal, init, ValueRange{j, k});
      }
    }
    rewriter.create<memref::DeallocOp>(loc, scratch);

    rewriter.eraseOp(reduceOp);
    return success();
  }

  return rewriter.notifyMatchFailure(
      reduceOp,
      "only rank-1 -> rank-0 (dimensions=[0]) and rank-2 -> rank-1 "
      "(dimensions=[1] or static-shape dimensions=[0]) plus static-shape "
      "rank-3 -> rank-2 (dimensions=[0]) reductions are supported");
}

static LogicalResult
lowerIndexOnlyGenericToScalarLoop(linalg::LinalgOp linalgOp,
                                  PatternRewriter &rewriter) {
  auto genericOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation());
  if (!genericOp)
    return rewriter.notifyMatchFailure(
        linalgOp, "index-only fallback expects linalg.generic");
  if (genericOp.getNumLoops() != 1)
    return rewriter.notifyMatchFailure(genericOp,
                                       "only 1D index-only generic supported");
  auto itTy = genericOp.getIteratorTypesArray()[0];
  if (!linalg::isParallelIterator(itTy))
    return rewriter.notifyMatchFailure(genericOp,
                                       "only parallel iterator supported");
  if (genericOp.getNumDpsInputs() != 0 || genericOp.getNumDpsInits() != 1)
    return rewriter.notifyMatchFailure(
        genericOp, "only outs-only single-output generic supported");

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
             arith::TruncFOp, arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
             arith::FPToUIOp, arith::BitcastOp, arith::IndexCastOp,
             arith::IndexCastUIOp>(op);
}

static bool isVectorizableFloatCastOp(Operation &op) {
  return isa<arith::ExtFOp, arith::TruncFOp>(op);
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
        return rewriter.notifyMatchFailure(
            genericOp, "unsupported operand in cast fallback");
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
             arith::MinimumFOp, arith::MaxSIOp, arith::MinSIOp, arith::MaxUIOp,
             arith::MinUIOp, arith::MaxNumFOp, arith::MinNumFOp, math::ExpOp,
             math::LogOp, math::AbsFOp, math::CeilOp, math::FloorOp,
             math::RoundOp, math::SqrtOp, math::RsqrtOp, math::TanhOp,
             math::ErfOp, math::PowFOp>(inner);
}

struct StoreOnlyGenericInfo {
  memref::StoreOp storeOp;
  Value storedValue;
  Value storedValueSource;
  Value scatterIndexSource;
  Value maskValue;
  bool isContinuous = false;
  bool isScatter = false;
  bool hasMask = false;
};

static Value stripSupportedIndexCast(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    if (auto castOp = dyn_cast<arith::IndexCastOp>(def)) {
      value = castOp.getIn();
      continue;
    }
    if (auto castOp = dyn_cast<arith::IndexCastUIOp>(def)) {
      value = castOp.getIn();
      continue;
    }
    break;
  }
  return value;
}

static OpOperand *findOperandForBlockArgument(linalg::GenericOp genericOp,
                                              Value value) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg || blockArg.getOwner() != genericOp.getBlock())
    return nullptr;
  for (OpOperand *opOperand : genericOp.getOpOperandsMatchingBBargs()) {
    if (genericOp.getMatchingBlockArgument(opOperand) == blockArg)
      return opOperand;
  }
  return nullptr;
}

static bool isIntegerOrIndexElement(Type type) {
  return type.isIndex() || isa<IntegerType>(type);
}

static bool isSupportedStoreOnlyIndexOp(Operation &op) {
  return isa<linalg::IndexOp, arith::IndexCastOp, arith::IndexCastUIOp>(op);
}

static FailureOr<StoreOnlyGenericInfo>
matchVectorizableStoreOnlyGeneric(linalg::GenericOp genericOp) {
  if (!genericOp.hasPureBufferSemantics())
    return failure();
  if (genericOp.getNumDpsInits() != 0)
    return failure();
  if (!llvm::all_of(genericOp.getIteratorTypesArray(),
                    [](utils::IteratorType itTy) {
                      return linalg::isParallelIterator(itTy);
                    }))
    return failure();

  Block &body = genericOp.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 0)
    return failure();

  memref::StoreOp storeOp;
  memref::LoadOp oldValueLoadOp;
  arith::SelectOp maskSelectOp;
  SmallVector<Operation *> storedValueCastCandidates;
  for (Operation &inner : body.without_terminator()) {
    if (auto candidate = dyn_cast<memref::StoreOp>(inner)) {
      if (storeOp)
        return failure();
      storeOp = candidate;
      continue;
    }
    if (auto candidate = dyn_cast<memref::LoadOp>(inner)) {
      if (oldValueLoadOp)
        return failure();
      oldValueLoadOp = candidate;
      continue;
    }
    if (auto candidate = dyn_cast<arith::SelectOp>(inner)) {
      if (maskSelectOp)
        return failure();
      maskSelectOp = candidate;
      continue;
    }
    if (isSupportedStoreOnlyIndexOp(inner))
      continue;
    if (isVectorizableFloatCastOp(inner)) {
      storedValueCastCandidates.push_back(&inner);
      continue;
    }
    return failure();
  }
  if (!storeOp)
    return failure();

  auto storeType = dyn_cast<MemRefType>(storeOp.getMemRef().getType());
  if (!storeType || storeType.getRank() !=
                        static_cast<int64_t>(storeOp.getIndices().size()))
    return failure();
  if (storeOp.getValueToStore().getType() != storeType.getElementType())
    return failure();

  StoreOnlyGenericInfo info;
  info.storeOp = storeOp;
  info.storedValue = storeOp.getValueToStore();
  if (auto selectOp = info.storedValue.getDefiningOp<arith::SelectOp>()) {
    if (selectOp != maskSelectOp || !oldValueLoadOp)
      return failure();
    if (selectOp.getFalseValue() != oldValueLoadOp.getResult())
      return failure();
    if (oldValueLoadOp.getMemRef() != storeOp.getMemRef())
      return failure();
    if (oldValueLoadOp.getIndices().size() != storeOp.getIndices().size())
      return failure();
    for (auto [loadIdx, storeIdx] :
         llvm::zip(oldValueLoadOp.getIndices(), storeOp.getIndices())) {
      if (loadIdx != storeIdx)
        return failure();
    }
    if (!selectOp.getCondition().getType().isInteger(1))
      return failure();
    info.storedValue = selectOp.getTrueValue();
    info.maskValue = selectOp.getCondition();
    info.hasMask = true;
  } else if (maskSelectOp || oldValueLoadOp) {
    return failure();
  }
  info.storedValueSource = info.storedValue;
  Operation *storedValueDef = info.storedValue.getDefiningOp();
  if (storedValueDef && isVectorizableFloatCastOp(*storedValueDef) &&
      storedValueDef->getNumOperands() == 1) {
    info.storedValueSource = storedValueDef->getOperand(0);
  }
  for (Operation *candidate : storedValueCastCandidates) {
    if (candidate != storedValueDef)
      return failure();
  }
  OpOperand *storedValueOperand =
      findOperandForBlockArgument(genericOp, info.storedValueSource);
  if (!storedValueOperand)
    return failure();
  if (info.hasMask && !findOperandForBlockArgument(genericOp, info.maskValue))
    return failure();

  bool continuous = storeOp.getIndices().size() == genericOp.getNumLoops();
  if (continuous) {
    for (auto [dim, index] : llvm::enumerate(storeOp.getIndices())) {
      auto indexOp = index.getDefiningOp<linalg::IndexOp>();
      if (!indexOp || indexOp.getDim() != dim) {
        continuous = false;
        break;
      }
    }
  }
  if (continuous) {
    info.isContinuous = true;
    return info;
  }

  if (storeOp.getIndices().size() == 1) {
    Value indexSource = stripSupportedIndexCast(storeOp.getIndices().front());
    OpOperand *indexOperand =
        findOperandForBlockArgument(genericOp, indexSource);
    if (!indexOperand || genericOp.isScalar(indexOperand))
      return failure();
    if (!isIntegerOrIndexElement(indexSource.getType()))
      return failure();
    info.scatterIndexSource = indexSource;
    info.isScatter = true;
    return info;
  }

  return failure();
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

static bool isSupportedMatmulElementTypes(Type inputElemTy, Type accElemTy) {
  if (inputElemTy == accElemTy)
    return isa<FloatType, IntegerType>(inputElemTy);

  if (auto inputFloat = dyn_cast<FloatType>(inputElemTy)) {
    auto accFloat = dyn_cast<FloatType>(accElemTy);
    return accFloat && inputFloat.getWidth() < accFloat.getWidth();
  }

  if (auto inputInt = dyn_cast<IntegerType>(inputElemTy)) {
    auto accInt = dyn_cast<IntegerType>(accElemTy);
    return accInt && inputInt.getWidth() < accInt.getWidth();
  }

  return false;
}

static bool isIntegerAccumulator(Type accElemTy) {
  return isa<IntegerType>(accElemTy);
}

static Value castVectorToAccumulatorType(
    OpBuilder &builder, Location loc, Value value,
    buddy::vir::DynamicVectorType accVecTy, linalg::TypeFn castFn) {
  auto dynTy = dyn_cast<buddy::vir::DynamicVectorType>(value.getType());
  if (!dynTy || dynTy.getElementType() == accVecTy.getElementType())
    return value;

  Type inputElemTy = dynTy.getElementType();
  Type accElemTy = accVecTy.getElementType();
  if (isa<FloatType>(inputElemTy) && isa<FloatType>(accElemTy))
    return builder.create<buddy::vir::ExtFOp>(loc, accVecTy, value)
        .getResult();

  if (isa<IntegerType>(inputElemTy) && isa<IntegerType>(accElemTy)) {
    if (castFn == linalg::TypeFn::cast_unsigned)
      return builder.create<buddy::vir::ExtUIOp>(loc, accVecTy, value)
          .getResult();
    return builder.create<buddy::vir::ExtSIOp>(loc, accVecTy, value)
        .getResult();
  }

  return Value();
}

static Value createMatmulAccumulation(
    OpBuilder &builder, Location loc, Value lhs, Value rhs, Value acc,
    buddy::vir::DynamicVectorType accVecTy) {
  Type accElemTy = accVecTy.getElementType();
  if (isa<FloatType>(accElemTy))
    return builder.create<buddy::vir::FMAOp>(loc, accVecTy, lhs, rhs, acc)
        .getResult();

  assert(isIntegerAccumulator(accElemTy) &&
         "expected integer accumulator for integer matmul");
  Value product = builder.create<arith::MulIOp>(loc, lhs, rhs).getResult();
  return builder.create<arith::AddIOp>(loc, product, acc).getResult();
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

  Type inputElemTy = aTy.getElementType();
  Type accElemTy = cTy.getElementType();
  if (inputElemTy != bTy.getElementType())
    return rewriter.notifyMatchFailure(matmulOp,
                                       "input element types must match");
  if (!isSupportedMatmulElementTypes(inputElemTy, accElemTy))
    return rewriter.notifyMatchFailure(matmulOp,
                                       "unsupported matmul element types");

  // Vectorize along N (the last dimension of B/C).
  Value n = rewriter.create<memref::DimOp>(loc, c, 1);

  // Create vir.set_vl region to host vector code.
  buddy::vir::SetVLOp setVl = createSetVLRegion(rewriter, loc, n);

  // Constants used inside the region.
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

  // Determine M/K. Prefer static when available, otherwise take dim from
  // memref.
  Value mVal;
  if (!ShapedType::isDynamic(cTy.getShape()[0])) {
    mVal = rewriter.create<arith::ConstantIndexOp>(loc, cTy.getShape()[0]);
  } else {
    mVal = rewriter.create<memref::DimOp>(loc, c, 0);
  }

  Value kVal;
  if (!ShapedType::isDynamic(aTy.getShape()[1])) {
    kVal = rewriter.create<arith::ConstantIndexOp>(loc, aTy.getShape()[1]);
  } else {
    kVal = rewriter.create<memref::DimOp>(loc, a, 1);
  }

  auto inputVecTy =
      buddy::vir::DynamicVectorType::get({ShapedType::kDynamic}, inputElemTy);
  auto accVecTy =
      buddy::vir::DynamicVectorType::get({ShapedType::kDynamic}, accElemTy);
  linalg::TypeFn castFn = matmulOp.getCast();

  auto loopM = rewriter.create<affine::AffineForOp>(
      loc, ValueRange{c0}, rewriter.getDimIdentityMap(), ValueRange{mVal},
      rewriter.getDimIdentityMap(), /*step=*/1,
      /*iterArgs=*/ValueRange{},
      [&](OpBuilder &bld, Location bodyLoc, Value i, ValueRange) {
        OpBuilder &builder = bld;
        // acc = load(C[i, 0:]) as a vector along N.
        Value acc = builder
                        .create<buddy::vir::LoadOp>(bodyLoc, accVecTy, c,
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
              Value aVec = builderK
                               .create<buddy::vir::BroadcastOp>(kLoc,
                                                                 inputVecTy,
                                                                 aScalar)
                               .getResult();
              // bVec = load(B[k, 0:]) as a vector along N.
              Value bVec = builderK
                               .create<buddy::vir::LoadOp>(kLoc, inputVecTy, b,
                                                           ValueRange{k, c0})
                               .getResult();
              Value aAccVec = castVectorToAccumulatorType(
                  builderK, kLoc, aVec, accVecTy, castFn);
              Value bAccVec = castVectorToAccumulatorType(
                  builderK, kLoc, bVec, accVecTy, castFn);
              if (!aAccVec || !bAccVec) {
                matmulOp.emitError("failed to widen matmul inputs");
                return;
              }
              Value accOut = createMatmulAccumulation(
                  builderK, kLoc, aAccVec, bAccVec, accIn, accVecTy);
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

static Value dimOrConstant(OpBuilder &builder, Location loc, Value memref,
                           int64_t dim) {
  auto ty = cast<MemRefType>(memref.getType());
  int64_t staticDim = ty.getShape()[dim];
  if (!ShapedType::isDynamic(staticDim))
    return builder.create<arith::ConstantIndexOp>(loc, staticDim);
  return builder.create<memref::DimOp>(loc, memref, dim);
}

static SmallVector<Value> makeAffineMapResults(OpBuilder &builder, Location loc,
                                               AffineMap map, ValueRange ivs) {
  SmallVector<Value> results;
  SmallVector<OpFoldResult> ofrs;
  ofrs.reserve(ivs.size());
  for (Value iv : ivs)
    ofrs.push_back(iv);
  results.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    auto singleResultMap =
        AffineMap::get(map.getNumDims(), map.getNumSymbols(), expr);
    results.push_back(
        affine::makeComposedAffineApply(builder, loc, singleResultMap, ofrs));
  }
  return results;
}

static LogicalResult touchFirstOutputWithVIR(Operation *sourceOp,
                                             PatternRewriter &rewriter,
                                             Value output) {
  auto outTy = dyn_cast<MemRefType>(output.getType());
  if (!outTy || outTy.getRank() == 0)
    return success();

  Location loc = sourceOp->getLoc();
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value vl = dimOrConstant(rewriter, loc, output, outTy.getRank() - 1);
  buddy::vir::SetVLOp setVl = createSetVLRegion(rewriter, loc, vl);

  SmallVector<Value> indices(outTy.getRank(), c0);
  auto emitAtDepth = [&](auto &&emitAtDepth, unsigned depth) -> void {
    if (depth == static_cast<unsigned>(outTy.getRank() - 1)) {
      auto vecTy = buddy::vir::DynamicVectorType::get({ShapedType::kDynamic},
                                                      outTy.getElementType());
      Value vec =
          rewriter.create<buddy::vir::LoadOp>(loc, vecTy, output, indices)
              .getResult();
      rewriter.create<buddy::vir::StoreOp>(loc, vec, output, indices);
      return;
    }
    Value upper = dimOrConstant(rewriter, loc, output, depth);
    auto loop = rewriter.create<scf::ForOp>(loc, c0, upper, c1);
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      indices[depth] = loop.getInductionVar();
      emitAtDepth(emitAtDepth, depth + 1);
    }
  };
  emitAtDepth(emitAtDepth, 0);
  rewriter.create<vector::YieldOp>(loc);
  rewriter.setInsertionPointAfter(setVl);
  return success();
}

static LogicalResult
lowerGenericToScalarLoopsWithVIRMarker(linalg::GenericOp genericOp,
                                       PatternRewriter &rewriter) {
  if (!genericOp.hasPureBufferSemantics())
    return rewriter.notifyMatchFailure(genericOp,
                                       "expected pure buffer semantics");
  if (genericOp.getNumDpsInits() == 0)
    return rewriter.notifyMatchFailure(genericOp, "expected an output");

  Location loc = genericOp.getLoc();
  SmallVector<Value> lbs;
  SmallVector<Value> ubs;
  SmallVector<Value> steps;
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  for (int64_t i = 0, e = genericOp.getNumLoops(); i < e; ++i) {
    lbs.push_back(c0);
    steps.push_back(c1);
    int64_t staticSize = genericOp.getStaticLoopRanges()[i];
    if (!ShapedType::isDynamic(staticSize)) {
      ubs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, staticSize));
      continue;
    }
    Value operand;
    unsigned operandDimPos = 0;
    if (failed(genericOp.mapIterationSpaceDimToOperandDim(i, operand,
                                                          operandDimPos)))
      return rewriter.notifyMatchFailure(genericOp,
                                         "cannot derive dynamic loop bound");
    ubs.push_back(rewriter.create<memref::DimOp>(loc, operand, operandDimPos));
  }

  rewriter.setInsertionPoint(genericOp);
  SmallVector<Value> ivs;
  auto emitLoopNest = [&](auto &&emitLoopNest,
                          unsigned depth) -> LogicalResult {
    if (depth == genericOp.getNumLoops()) {
      IRMapping mapper;
      for (OpOperand *opOperand : genericOp.getOpOperandsMatchingBBargs()) {
        BlockArgument bbArg = genericOp.getMatchingBlockArgument(opOperand);
        if (genericOp.isScalar(opOperand)) {
          mapper.map(bbArg, opOperand->get());
          continue;
        }
        AffineMap map = genericOp.getMatchingIndexingMap(opOperand);
        SmallVector<Value> indices =
            makeAffineMapResults(rewriter, loc, map, ivs);
        Value loaded =
            rewriter.create<memref::LoadOp>(loc, opOperand->get(), indices);
        mapper.map(bbArg, loaded);
      }

      Block &body = genericOp.getRegion().front();
      auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
      if (!yield)
        return rewriter.notifyMatchFailure(genericOp, "expected linalg.yield");

      for (Operation &inner : body.without_terminator()) {
        if (auto indexOp = dyn_cast<linalg::IndexOp>(inner)) {
          mapper.map(indexOp.getResult(), ivs[indexOp.getDim()]);
          continue;
        }
        rewriter.clone(inner, mapper);
      }

      for (auto [idx, yielded] : llvm::enumerate(yield.getOperands())) {
        OpOperand *init = genericOp.getDpsInitOperand(idx);
        AffineMap map = genericOp.getMatchingIndexingMap(init);
        SmallVector<Value> indices =
            makeAffineMapResults(rewriter, loc, map, ivs);
        rewriter.create<memref::StoreOp>(loc, mapper.lookup(yielded),
                                         init->get(), indices);
      }
      return success();
    }

    auto loop =
        rewriter.create<scf::ForOp>(loc, lbs[depth], ubs[depth], steps[depth]);
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      ivs.push_back(loop.getInductionVar());
      if (failed(emitLoopNest(emitLoopNest, depth + 1)))
        return failure();
      ivs.pop_back();
    }
    return success();
  };

  if (failed(emitLoopNest(emitLoopNest, 0)))
    return failure();
  if (failed(touchFirstOutputWithVIR(genericOp.getOperation(), rewriter,
                                     genericOp.getDpsInitOperand(0)->get())))
    return failure();
  rewriter.eraseOp(genericOp);
  return success();
}

static LogicalResult
lowerRank2MatmulLikeToVIR(Operation *op, PatternRewriter &rewriter, Value a,
                          Value b, Value c, bool transposeA, bool transposeB) {
  Location loc = op->getLoc();
  auto aTy = dyn_cast<MemRefType>(a.getType());
  auto bTy = dyn_cast<MemRefType>(b.getType());
  auto cTy = dyn_cast<MemRefType>(c.getType());
  if (!aTy || !bTy || !cTy || aTy.getRank() != 2 || bTy.getRank() != 2 ||
      cTy.getRank() != 2)
    return rewriter.notifyMatchFailure(op, "expected rank-2 memrefs");

  Type elemTy = aTy.getElementType();
  if (elemTy != bTy.getElementType() || elemTy != cTy.getElementType() ||
      !isa<FloatType>(elemTy))
    return rewriter.notifyMatchFailure(
        op, "expected matching floating-point element types");

  Value n = dimOrConstant(rewriter, loc, c, 1);
  buddy::vir::SetVLOp setVl = createSetVLRegion(rewriter, loc, n);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value mVal = dimOrConstant(rewriter, loc, c, 0);
  Value kVal = dimOrConstant(rewriter, loc, a, transposeA ? 0 : 1);

  Value bForLoad = b;
  if (transposeB) {
    SmallVector<int64_t> permutation = {1, 0};
    auto permutationMap =
        AffineMap::getPermutationMap(permutation, rewriter.getContext());
    bForLoad = rewriter.create<memref::TransposeOp>(
        loc, b, AffineMapAttr::get(permutationMap));
  }

  auto vecTy =
      buddy::vir::DynamicVectorType::get({ShapedType::kDynamic}, elemTy);
  rewriter.create<affine::AffineForOp>(
      loc, ValueRange{c0}, rewriter.getDimIdentityMap(), ValueRange{mVal},
      rewriter.getDimIdentityMap(), /*step=*/1, /*iterArgs=*/ValueRange{},
      [&](OpBuilder &bld, Location bodyLoc, Value i, ValueRange) {
        Value acc =
            bld.create<buddy::vir::LoadOp>(bodyLoc, vecTy, c, ValueRange{i, c0})
                .getResult();
        auto loopK = bld.create<affine::AffineForOp>(
            bodyLoc, ValueRange{c0}, rewriter.getDimIdentityMap(),
            ValueRange{kVal}, rewriter.getDimIdentityMap(), /*step=*/1,
            /*iterArgs=*/ValueRange{acc},
            [&](OpBuilder &kb, Location kLoc, Value k, ValueRange iterArgs) {
              Value aScalar =
                  transposeA
                      ? kb.create<memref::LoadOp>(kLoc, a, ValueRange{k, i})
                      : kb.create<memref::LoadOp>(kLoc, a, ValueRange{i, k});
              Value aVec =
                  kb.create<buddy::vir::BroadcastOp>(kLoc, vecTy, aScalar)
                      .getResult();
              Value bVec = kb.create<buddy::vir::LoadOp>(kLoc, vecTy, bForLoad,
                                                         ValueRange{k, c0})
                               .getResult();
              Value accOut = kb.create<buddy::vir::FMAOp>(kLoc, vecTy, aVec,
                                                          bVec, iterArgs[0])
                                 .getResult();
              kb.create<affine::AffineYieldOp>(kLoc, accOut);
            });
        bld.create<buddy::vir::StoreOp>(bodyLoc, loopK.getResult(0), c,
                                        ValueRange{i, c0});
        bld.create<affine::AffineYieldOp>(bodyLoc);
      });
  rewriter.create<vector::YieldOp>(loc);
  rewriter.replaceOp(op, setVl.getResults());
  return success();
}

static LogicalResult
lowerBatchMatmulLikeToVIR(Operation *op, PatternRewriter &rewriter, Value a,
                          Value b, Value c, bool transposeA, bool transposeB) {
  Location loc = op->getLoc();
  auto aTy = dyn_cast<MemRefType>(a.getType());
  auto bTy = dyn_cast<MemRefType>(b.getType());
  auto cTy = dyn_cast<MemRefType>(c.getType());
  if (!aTy || !bTy || !cTy || aTy.getRank() != 3 || bTy.getRank() != 3 ||
      cTy.getRank() != 3)
    return rewriter.notifyMatchFailure(op, "expected rank-3 memrefs");

  Type elemTy = aTy.getElementType();
  if (elemTy != bTy.getElementType() || elemTy != cTy.getElementType() ||
      !isa<FloatType>(elemTy))
    return rewriter.notifyMatchFailure(
        op, "expected matching floating-point element types");

  Value n = dimOrConstant(rewriter, loc, c, 2);
  buddy::vir::SetVLOp setVl = createSetVLRegion(rewriter, loc, n);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value batchVal = dimOrConstant(rewriter, loc, c, 0);
  Value mVal = dimOrConstant(rewriter, loc, c, 1);
  Value kVal = dimOrConstant(rewriter, loc, a, transposeA ? 1 : 2);

  Value bForLoad = b;
  if (transposeB) {
    SmallVector<int64_t> permutation = {0, 2, 1};
    auto permutationMap =
        AffineMap::getPermutationMap(permutation, rewriter.getContext());
    bForLoad = rewriter.create<memref::TransposeOp>(
        loc, b, AffineMapAttr::get(permutationMap));
  }

  auto vecTy =
      buddy::vir::DynamicVectorType::get({ShapedType::kDynamic}, elemTy);
  rewriter.create<affine::AffineForOp>(
      loc, ValueRange{c0}, rewriter.getDimIdentityMap(), ValueRange{batchVal},
      rewriter.getDimIdentityMap(), /*step=*/1, /*iterArgs=*/ValueRange{},
      [&](OpBuilder &bb, Location bLoc, Value batch, ValueRange) {
        bb.create<affine::AffineForOp>(
            bLoc, ValueRange{c0}, rewriter.getDimIdentityMap(),
            ValueRange{mVal}, rewriter.getDimIdentityMap(), /*step=*/1,
            /*iterArgs=*/ValueRange{},
            [&](OpBuilder &mb, Location mLoc, Value i, ValueRange) {
              Value acc = mb.create<buddy::vir::LoadOp>(
                                mLoc, vecTy, c, ValueRange{batch, i, c0})
                              .getResult();
              auto loopK = mb.create<affine::AffineForOp>(
                  mLoc, ValueRange{c0}, rewriter.getDimIdentityMap(),
                  ValueRange{kVal}, rewriter.getDimIdentityMap(), /*step=*/1,
                  /*iterArgs=*/ValueRange{acc},
                  [&](OpBuilder &kb, Location kLoc, Value k,
                      ValueRange iterArgs) {
                    Value aScalar = transposeA
                                        ? kb.create<memref::LoadOp>(
                                              kLoc, a, ValueRange{batch, k, i})
                                        : kb.create<memref::LoadOp>(
                                              kLoc, a, ValueRange{batch, i, k});
                    Value aVec =
                        kb.create<buddy::vir::BroadcastOp>(kLoc, vecTy, aScalar)
                            .getResult();
                    Value bVec =
                        kb.create<buddy::vir::LoadOp>(kLoc, vecTy, bForLoad,
                                                      ValueRange{batch, k, c0})
                            .getResult();
                    Value accOut = kb.create<buddy::vir::FMAOp>(
                                         kLoc, vecTy, aVec, bVec, iterArgs[0])
                                       .getResult();
                    kb.create<affine::AffineYieldOp>(kLoc, accOut);
                  });
              mb.create<buddy::vir::StoreOp>(mLoc, loopK.getResult(0), c,
                                             ValueRange{batch, i, c0});
              mb.create<affine::AffineYieldOp>(mLoc);
            });
        bb.create<affine::AffineYieldOp>(bLoc);
      });
  rewriter.create<vector::YieldOp>(loc);
  rewriter.replaceOp(op, setVl.getResults());
  return success();
}

static LogicalResult lowerVecmatToVIR(linalg::VecmatOp op,
                                      PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value x = op.getInputs()[0];
  Value b = op.getInputs()[1];
  Value y = op.getOutputs()[0];
  auto xTy = dyn_cast<MemRefType>(x.getType());
  auto bTy = dyn_cast<MemRefType>(b.getType());
  auto yTy = dyn_cast<MemRefType>(y.getType());
  if (!xTy || !bTy || !yTy || xTy.getRank() != 1 || bTy.getRank() != 2 ||
      yTy.getRank() != 1)
    return rewriter.notifyMatchFailure(op, "expected vecmat memrefs");
  Type elemTy = yTy.getElementType();
  Value n = dimOrConstant(rewriter, loc, y, 0);
  Value kVal = dimOrConstant(rewriter, loc, x, 0);
  buddy::vir::SetVLOp setVl = createSetVLRegion(rewriter, loc, n);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto vecTy =
      buddy::vir::DynamicVectorType::get({ShapedType::kDynamic}, elemTy);
  Value acc = rewriter.create<buddy::vir::LoadOp>(loc, vecTy, y, ValueRange{c0})
                  .getResult();
  auto loopK = rewriter.create<affine::AffineForOp>(
      loc, ValueRange{c0}, rewriter.getDimIdentityMap(), ValueRange{kVal},
      rewriter.getDimIdentityMap(), /*step=*/1, /*iterArgs=*/ValueRange{acc},
      [&](OpBuilder &kb, Location kLoc, Value k, ValueRange iterArgs) {
        Value xScalar = kb.create<memref::LoadOp>(kLoc, x, ValueRange{k});
        Value xVec = kb.create<buddy::vir::BroadcastOp>(kLoc, vecTy, xScalar)
                         .getResult();
        Value bVec =
            kb.create<buddy::vir::LoadOp>(kLoc, vecTy, b, ValueRange{k, c0})
                .getResult();
        Value accOut =
            kb.create<buddy::vir::FMAOp>(kLoc, vecTy, xVec, bVec, iterArgs[0])
                .getResult();
        kb.create<affine::AffineYieldOp>(kLoc, accOut);
      });
  rewriter.create<buddy::vir::StoreOp>(loc, loopK.getResult(0), y,
                                       ValueRange{c0});
  rewriter.create<vector::YieldOp>(loc);
  rewriter.replaceOp(op, setVl.getResults());
  return success();
}

static LogicalResult lowerMatvecToVIR(linalg::MatvecOp op,
                                      PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value a = op.getInputs()[0];
  Value x = op.getInputs()[1];
  Value y = op.getOutputs()[0];
  auto aTy = dyn_cast<MemRefType>(a.getType());
  auto xTy = dyn_cast<MemRefType>(x.getType());
  auto yTy = dyn_cast<MemRefType>(y.getType());
  if (!aTy || !xTy || !yTy || aTy.getRank() != 2 || xTy.getRank() != 1 ||
      yTy.getRank() != 1)
    return rewriter.notifyMatchFailure(op, "expected matvec memrefs");
  Type elemTy = yTy.getElementType();
  Value m = dimOrConstant(rewriter, loc, y, 0);
  Value kVal = dimOrConstant(rewriter, loc, x, 0);
  buddy::vir::SetVLOp setVl = createSetVLRegion(rewriter, loc, m);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<int64_t> permutation = {1, 0};
  auto permutationMap =
      AffineMap::getPermutationMap(permutation, rewriter.getContext());
  Value aT = rewriter.create<memref::TransposeOp>(
      loc, a, AffineMapAttr::get(permutationMap));
  auto vecTy =
      buddy::vir::DynamicVectorType::get({ShapedType::kDynamic}, elemTy);
  Value acc = rewriter.create<buddy::vir::LoadOp>(loc, vecTy, y, ValueRange{c0})
                  .getResult();
  auto loopK = rewriter.create<affine::AffineForOp>(
      loc, ValueRange{c0}, rewriter.getDimIdentityMap(), ValueRange{kVal},
      rewriter.getDimIdentityMap(), /*step=*/1, /*iterArgs=*/ValueRange{acc},
      [&](OpBuilder &kb, Location kLoc, Value k, ValueRange iterArgs) {
        Value xScalar = kb.create<memref::LoadOp>(kLoc, x, ValueRange{k});
        Value xVec = kb.create<buddy::vir::BroadcastOp>(kLoc, vecTy, xScalar)
                         .getResult();
        Value aVec =
            kb.create<buddy::vir::LoadOp>(kLoc, vecTy, aT, ValueRange{k, c0})
                .getResult();
        Value accOut =
            kb.create<buddy::vir::FMAOp>(kLoc, vecTy, aVec, xVec, iterArgs[0])
                .getResult();
        kb.create<affine::AffineYieldOp>(kLoc, accOut);
      });
  rewriter.create<buddy::vir::StoreOp>(loc, loopK.getResult(0), y,
                                       ValueRange{c0});
  rewriter.create<vector::YieldOp>(loc);
  rewriter.replaceOp(op, setVl.getResults());
  return success();
}

static LogicalResult lowerDotToVIR(linalg::DotOp op,
                                   PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value a = op.getInputs()[0];
  Value b = op.getInputs()[1];
  Value c = op.getOutputs()[0];
  auto aTy = dyn_cast<MemRefType>(a.getType());
  auto bTy = dyn_cast<MemRefType>(b.getType());
  auto cTy = dyn_cast<MemRefType>(c.getType());
  if (!aTy || !bTy || !cTy || aTy.getRank() != 1 || bTy.getRank() != 1 ||
      cTy.getRank() != 0)
    return rewriter.notifyMatchFailure(op, "expected dot memrefs");
  Type elemTy = cTy.getElementType();
  Value kVal = dimOrConstant(rewriter, loc, a, 0);
  buddy::vir::SetVLOp setVl = createSetVLRegion(rewriter, loc, kVal);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto vecTy =
      buddy::vir::DynamicVectorType::get({ShapedType::kDynamic}, elemTy);
  Value aVec =
      rewriter.create<buddy::vir::LoadOp>(loc, vecTy, a, ValueRange{c0})
          .getResult();
  Value bVec =
      rewriter.create<buddy::vir::LoadOp>(loc, vecTy, b, ValueRange{c0})
          .getResult();
  Value prod = rewriter.create<arith::MulFOp>(loc, aVec, bVec).getResult();
  Value acc = rewriter.create<memref::LoadOp>(loc, c, ValueRange{});
  Value reduced = rewriter
                      .create<buddy::vir::ReduceOp>(
                          loc, elemTy, prod, acc, rewriter.getStringAttr("add"))
                      .getResult();
  rewriter.create<memref::StoreOp>(loc, reduced, c, ValueRange{});
  rewriter.create<vector::YieldOp>(loc);
  rewriter.replaceOp(op, setVl.getResults());
  return success();
}

static LogicalResult lowerBatchMatvecToVIR(linalg::BatchMatvecOp op,
                                           PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value a = op.getInputs()[0];
  Value x = op.getInputs()[1];
  Value y = op.getOutputs()[0];
  auto aTy = dyn_cast<MemRefType>(a.getType());
  auto xTy = dyn_cast<MemRefType>(x.getType());
  auto yTy = dyn_cast<MemRefType>(y.getType());
  if (!aTy || !xTy || !yTy || aTy.getRank() != 3 || xTy.getRank() != 2 ||
      yTy.getRank() != 2)
    return rewriter.notifyMatchFailure(op, "expected batch_matvec memrefs");

  Type elemTy = yTy.getElementType();
  if (elemTy != aTy.getElementType() || elemTy != xTy.getElementType() ||
      !isa<FloatType>(elemTy))
    return rewriter.notifyMatchFailure(
        op, "expected matching floating-point element types");

  Value m = dimOrConstant(rewriter, loc, y, 1);
  buddy::vir::SetVLOp setVl = createSetVLRegion(rewriter, loc, m);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value batchVal = dimOrConstant(rewriter, loc, y, 0);
  Value kVal = dimOrConstant(rewriter, loc, x, 1);

  SmallVector<int64_t> permutation = {0, 2, 1};
  auto permutationMap =
      AffineMap::getPermutationMap(permutation, rewriter.getContext());
  Value aT = rewriter.create<memref::TransposeOp>(
      loc, a, AffineMapAttr::get(permutationMap));

  auto vecTy =
      buddy::vir::DynamicVectorType::get({ShapedType::kDynamic}, elemTy);
  rewriter.create<affine::AffineForOp>(
      loc, ValueRange{c0}, rewriter.getDimIdentityMap(), ValueRange{batchVal},
      rewriter.getDimIdentityMap(), /*step=*/1, /*iterArgs=*/ValueRange{},
      [&](OpBuilder &bb, Location bLoc, Value batch, ValueRange) {
        Value acc =
            bb.create<buddy::vir::LoadOp>(bLoc, vecTy, y, ValueRange{batch, c0})
                .getResult();
        auto loopK = bb.create<affine::AffineForOp>(
            bLoc, ValueRange{c0}, rewriter.getDimIdentityMap(),
            ValueRange{kVal}, rewriter.getDimIdentityMap(), /*step=*/1,
            /*iterArgs=*/ValueRange{acc},
            [&](OpBuilder &kb, Location kLoc, Value k, ValueRange iterArgs) {
              Value xScalar =
                  kb.create<memref::LoadOp>(kLoc, x, ValueRange{batch, k});
              Value xVec =
                  kb.create<buddy::vir::BroadcastOp>(kLoc, vecTy, xScalar)
                      .getResult();
              Value aVec = kb.create<buddy::vir::LoadOp>(
                                 kLoc, vecTy, aT, ValueRange{batch, k, c0})
                               .getResult();
              Value accOut = kb.create<buddy::vir::FMAOp>(kLoc, vecTy, aVec,
                                                          xVec, iterArgs[0])
                                 .getResult();
              kb.create<affine::AffineYieldOp>(kLoc, accOut);
            });
        bb.create<buddy::vir::StoreOp>(bLoc, loopK.getResult(0), y,
                                       ValueRange{batch, c0});
        bb.create<affine::AffineYieldOp>(bLoc);
      });
  rewriter.create<vector::YieldOp>(loc);
  rewriter.replaceOp(op, setVl.getResults());
  return success();
}

static LogicalResult lowerBatchVecmatToVIR(linalg::BatchVecmatOp op,
                                           PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value x = op.getInputs()[0];
  Value b = op.getInputs()[1];
  Value y = op.getOutputs()[0];
  auto xTy = dyn_cast<MemRefType>(x.getType());
  auto bTy = dyn_cast<MemRefType>(b.getType());
  auto yTy = dyn_cast<MemRefType>(y.getType());
  if (!xTy || !bTy || !yTy || xTy.getRank() != 2 || bTy.getRank() != 3 ||
      yTy.getRank() != 2)
    return rewriter.notifyMatchFailure(op, "expected batch_vecmat memrefs");

  Type elemTy = yTy.getElementType();
  if (elemTy != xTy.getElementType() || elemTy != bTy.getElementType() ||
      !isa<FloatType>(elemTy))
    return rewriter.notifyMatchFailure(
        op, "expected matching floating-point element types");

  Value n = dimOrConstant(rewriter, loc, y, 1);
  buddy::vir::SetVLOp setVl = createSetVLRegion(rewriter, loc, n);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value batchVal = dimOrConstant(rewriter, loc, y, 0);
  Value kVal = dimOrConstant(rewriter, loc, x, 1);

  auto vecTy =
      buddy::vir::DynamicVectorType::get({ShapedType::kDynamic}, elemTy);
  rewriter.create<affine::AffineForOp>(
      loc, ValueRange{c0}, rewriter.getDimIdentityMap(), ValueRange{batchVal},
      rewriter.getDimIdentityMap(), /*step=*/1, /*iterArgs=*/ValueRange{},
      [&](OpBuilder &bb, Location bLoc, Value batch, ValueRange) {
        Value acc =
            bb.create<buddy::vir::LoadOp>(bLoc, vecTy, y, ValueRange{batch, c0})
                .getResult();
        auto loopK = bb.create<affine::AffineForOp>(
            bLoc, ValueRange{c0}, rewriter.getDimIdentityMap(),
            ValueRange{kVal}, rewriter.getDimIdentityMap(), /*step=*/1,
            /*iterArgs=*/ValueRange{acc},
            [&](OpBuilder &kb, Location kLoc, Value k, ValueRange iterArgs) {
              Value xScalar =
                  kb.create<memref::LoadOp>(kLoc, x, ValueRange{batch, k});
              Value xVec =
                  kb.create<buddy::vir::BroadcastOp>(kLoc, vecTy, xScalar)
                      .getResult();
              Value bVec = kb.create<buddy::vir::LoadOp>(
                                 kLoc, vecTy, b, ValueRange{batch, k, c0})
                               .getResult();
              Value accOut = kb.create<buddy::vir::FMAOp>(kLoc, vecTy, xVec,
                                                          bVec, iterArgs[0])
                                 .getResult();
              kb.create<affine::AffineYieldOp>(kLoc, accOut);
            });
        bb.create<buddy::vir::StoreOp>(bLoc, loopK.getResult(0), y,
                                       ValueRange{batch, c0});
        bb.create<affine::AffineYieldOp>(bLoc);
      });
  rewriter.create<vector::YieldOp>(loc);
  rewriter.replaceOp(op, setVl.getResults());
  return success();
}

static LogicalResult lowerBatchReduceMatmulToVIR(linalg::BatchReduceMatmulOp op,
                                                 PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value a = op.getInputs()[0];
  Value b = op.getInputs()[1];
  Value c = op.getOutputs()[0];
  auto aTy = dyn_cast<MemRefType>(a.getType());
  auto bTy = dyn_cast<MemRefType>(b.getType());
  auto cTy = dyn_cast<MemRefType>(c.getType());
  if (!aTy || !bTy || !cTy || aTy.getRank() != 3 || bTy.getRank() != 3 ||
      cTy.getRank() != 2)
    return rewriter.notifyMatchFailure(op,
                                       "expected batch_reduce_matmul memrefs");

  Type elemTy = cTy.getElementType();
  if (elemTy != aTy.getElementType() || elemTy != bTy.getElementType() ||
      !isa<FloatType>(elemTy))
    return rewriter.notifyMatchFailure(
        op, "expected matching floating-point element types");

  Value n = dimOrConstant(rewriter, loc, c, 1);
  buddy::vir::SetVLOp setVl = createSetVLRegion(rewriter, loc, n);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value batchVal = dimOrConstant(rewriter, loc, a, 0);
  Value mVal = dimOrConstant(rewriter, loc, c, 0);
  Value kVal = dimOrConstant(rewriter, loc, a, 2);

  auto vecTy =
      buddy::vir::DynamicVectorType::get({ShapedType::kDynamic}, elemTy);
  rewriter.create<affine::AffineForOp>(
      loc, ValueRange{c0}, rewriter.getDimIdentityMap(), ValueRange{mVal},
      rewriter.getDimIdentityMap(), /*step=*/1, /*iterArgs=*/ValueRange{},
      [&](OpBuilder &mb, Location mLoc, Value i, ValueRange) {
        Value acc =
            mb.create<buddy::vir::LoadOp>(mLoc, vecTy, c, ValueRange{i, c0})
                .getResult();
        auto loopBatch = mb.create<affine::AffineForOp>(
            mLoc, ValueRange{c0}, rewriter.getDimIdentityMap(),
            ValueRange{batchVal}, rewriter.getDimIdentityMap(), /*step=*/1,
            /*iterArgs=*/ValueRange{acc},
            [&](OpBuilder &bb, Location bLoc, Value batch,
                ValueRange batchIterArgs) {
              auto loopK = bb.create<affine::AffineForOp>(
                  bLoc, ValueRange{c0}, rewriter.getDimIdentityMap(),
                  ValueRange{kVal}, rewriter.getDimIdentityMap(), /*step=*/1,
                  /*iterArgs=*/ValueRange{batchIterArgs[0]},
                  [&](OpBuilder &kb, Location kLoc, Value k,
                      ValueRange iterArgs) {
                    Value aScalar = kb.create<memref::LoadOp>(
                        kLoc, a, ValueRange{batch, i, k});
                    Value aVec =
                        kb.create<buddy::vir::BroadcastOp>(kLoc, vecTy, aScalar)
                            .getResult();
                    Value bVec = kb.create<buddy::vir::LoadOp>(
                                       kLoc, vecTy, b, ValueRange{batch, k, c0})
                                     .getResult();
                    Value accOut = kb.create<buddy::vir::FMAOp>(
                                         kLoc, vecTy, aVec, bVec, iterArgs[0])
                                       .getResult();
                    kb.create<affine::AffineYieldOp>(kLoc, accOut);
                  });
              bb.create<affine::AffineYieldOp>(bLoc, loopK.getResult(0));
            });
        mb.create<buddy::vir::StoreOp>(mLoc, loopBatch.getResult(0), c,
                                       ValueRange{i, c0});
        mb.create<affine::AffineYieldOp>(mLoc);
      });
  rewriter.create<vector::YieldOp>(loc);
  rewriter.replaceOp(op, setVl.getResults());
  return success();
}

static LogicalResult lowerMmt4DToVIR(linalg::Mmt4DOp op,
                                     PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value a = op.getInputs()[0];
  Value b = op.getInputs()[1];
  Value c = op.getOutputs()[0];
  auto aTy = dyn_cast<MemRefType>(a.getType());
  auto bTy = dyn_cast<MemRefType>(b.getType());
  auto cTy = dyn_cast<MemRefType>(c.getType());
  if (!aTy || !bTy || !cTy || aTy.getRank() != 4 || bTy.getRank() != 4 ||
      cTy.getRank() != 4)
    return rewriter.notifyMatchFailure(op, "expected mmt4d memrefs");

  Type elemTy = cTy.getElementType();
  if (elemTy != aTy.getElementType() || elemTy != bTy.getElementType() ||
      !isa<FloatType>(elemTy))
    return rewriter.notifyMatchFailure(
        op, "expected matching floating-point element types");

  Value n1 = dimOrConstant(rewriter, loc, c, 3);
  buddy::vir::SetVLOp setVl = createSetVLRegion(rewriter, loc, n1);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value m0Val = dimOrConstant(rewriter, loc, c, 0);
  Value n0Val = dimOrConstant(rewriter, loc, c, 1);
  Value m1Val = dimOrConstant(rewriter, loc, c, 2);
  Value k0Val = dimOrConstant(rewriter, loc, a, 1);
  Value k1Val = dimOrConstant(rewriter, loc, a, 3);

  SmallVector<int64_t> permutation = {0, 1, 3, 2};
  auto permutationMap =
      AffineMap::getPermutationMap(permutation, rewriter.getContext());
  Value bT = rewriter.create<memref::TransposeOp>(
      loc, b, AffineMapAttr::get(permutationMap));

  auto vecTy =
      buddy::vir::DynamicVectorType::get({ShapedType::kDynamic}, elemTy);
  rewriter.create<affine::AffineForOp>(
      loc, ValueRange{c0}, rewriter.getDimIdentityMap(), ValueRange{m0Val},
      rewriter.getDimIdentityMap(), /*step=*/1, /*iterArgs=*/ValueRange{},
      [&](OpBuilder &m0b, Location m0Loc, Value m0, ValueRange) {
        m0b.create<affine::AffineForOp>(
            m0Loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
            ValueRange{n0Val}, rewriter.getDimIdentityMap(), /*step=*/1,
            /*iterArgs=*/ValueRange{},
            [&](OpBuilder &n0b, Location n0Loc, Value n0, ValueRange) {
              n0b.create<affine::AffineForOp>(
                  n0Loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
                  ValueRange{m1Val}, rewriter.getDimIdentityMap(), /*step=*/1,
                  /*iterArgs=*/ValueRange{},
                  [&](OpBuilder &m1b, Location m1Loc, Value m1, ValueRange) {
                    Value acc =
                        m1b.create<buddy::vir::LoadOp>(
                               m1Loc, vecTy, c, ValueRange{m0, n0, m1, c0})
                            .getResult();
                    auto loopK0 = m1b.create<affine::AffineForOp>(
                        m1Loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
                        ValueRange{k0Val}, rewriter.getDimIdentityMap(),
                        /*step=*/1, /*iterArgs=*/ValueRange{acc},
                        [&](OpBuilder &k0b, Location k0Loc, Value k0,
                            ValueRange k0IterArgs) {
                          auto loopK1 = k0b.create<affine::AffineForOp>(
                              k0Loc, ValueRange{c0},
                              rewriter.getDimIdentityMap(), ValueRange{k1Val},
                              rewriter.getDimIdentityMap(), /*step=*/1,
                              /*iterArgs=*/ValueRange{k0IterArgs[0]},
                              [&](OpBuilder &k1b, Location k1Loc, Value k1,
                                  ValueRange iterArgs) {
                                Value aScalar = k1b.create<memref::LoadOp>(
                                    k1Loc, a, ValueRange{m0, k0, m1, k1});
                                Value aVec =
                                    k1b.create<buddy::vir::BroadcastOp>(
                                           k1Loc, vecTy, aScalar)
                                        .getResult();
                                Value bVec = k1b.create<buddy::vir::LoadOp>(
                                                    k1Loc, vecTy, bT,
                                                    ValueRange{n0, k0, k1, c0})
                                                 .getResult();
                                Value accOut = k1b.create<buddy::vir::FMAOp>(
                                                      k1Loc, vecTy, aVec, bVec,
                                                      iterArgs[0])
                                                   .getResult();
                                k1b.create<affine::AffineYieldOp>(k1Loc,
                                                                  accOut);
                              });
                          k0b.create<affine::AffineYieldOp>(
                              k0Loc, loopK1.getResult(0));
                        });
                    m1b.create<buddy::vir::StoreOp>(m1Loc, loopK0.getResult(0),
                                                    c,
                                                    ValueRange{m0, n0, m1, c0});
                    m1b.create<affine::AffineYieldOp>(m1Loc);
                  });
              n0b.create<affine::AffineYieldOp>(n0Loc);
            });
        m0b.create<affine::AffineYieldOp>(m0Loc);
      });
  rewriter.create<vector::YieldOp>(loc);
  rewriter.replaceOp(op, setVl.getResults());
  return success();
}

static LogicalResult lowerBatchMmt4DToVIR(linalg::BatchMmt4DOp op,
                                          PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value a = op.getInputs()[0];
  Value b = op.getInputs()[1];
  Value c = op.getOutputs()[0];
  auto aTy = dyn_cast<MemRefType>(a.getType());
  auto bTy = dyn_cast<MemRefType>(b.getType());
  auto cTy = dyn_cast<MemRefType>(c.getType());
  if (!aTy || !bTy || !cTy || aTy.getRank() != 5 || bTy.getRank() != 5 ||
      cTy.getRank() != 5)
    return rewriter.notifyMatchFailure(op, "expected batch_mmt4d memrefs");

  Type elemTy = cTy.getElementType();
  if (elemTy != aTy.getElementType() || elemTy != bTy.getElementType() ||
      !isa<FloatType>(elemTy))
    return rewriter.notifyMatchFailure(
        op, "expected matching floating-point element types");

  Value n1 = dimOrConstant(rewriter, loc, c, 4);
  buddy::vir::SetVLOp setVl = createSetVLRegion(rewriter, loc, n1);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value batchVal = dimOrConstant(rewriter, loc, c, 0);
  Value m0Val = dimOrConstant(rewriter, loc, c, 1);
  Value n0Val = dimOrConstant(rewriter, loc, c, 2);
  Value m1Val = dimOrConstant(rewriter, loc, c, 3);
  Value k0Val = dimOrConstant(rewriter, loc, a, 2);
  Value k1Val = dimOrConstant(rewriter, loc, a, 4);

  SmallVector<int64_t> permutation = {0, 1, 2, 4, 3};
  auto permutationMap =
      AffineMap::getPermutationMap(permutation, rewriter.getContext());
  Value bT = rewriter.create<memref::TransposeOp>(
      loc, b, AffineMapAttr::get(permutationMap));

  auto vecTy =
      buddy::vir::DynamicVectorType::get({ShapedType::kDynamic}, elemTy);
  rewriter.create<affine::AffineForOp>(
      loc, ValueRange{c0}, rewriter.getDimIdentityMap(), ValueRange{batchVal},
      rewriter.getDimIdentityMap(), /*step=*/1, /*iterArgs=*/ValueRange{},
      [&](OpBuilder &bb, Location bLoc, Value batch, ValueRange) {
        bb.create<affine::AffineForOp>(
            bLoc, ValueRange{c0}, rewriter.getDimIdentityMap(),
            ValueRange{m0Val}, rewriter.getDimIdentityMap(), /*step=*/1,
            /*iterArgs=*/ValueRange{},
            [&](OpBuilder &m0b, Location m0Loc, Value m0, ValueRange) {
              m0b.create<affine::AffineForOp>(
                  m0Loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
                  ValueRange{n0Val}, rewriter.getDimIdentityMap(), /*step=*/1,
                  /*iterArgs=*/ValueRange{},
                  [&](OpBuilder &n0b, Location n0Loc, Value n0, ValueRange) {
                    n0b.create<affine::AffineForOp>(
                        n0Loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
                        ValueRange{m1Val}, rewriter.getDimIdentityMap(),
                        /*step=*/1, /*iterArgs=*/ValueRange{},
                        [&](OpBuilder &m1b, Location m1Loc, Value m1,
                            ValueRange) {
                          Value acc = m1b.create<buddy::vir::LoadOp>(
                                             m1Loc, vecTy, c,
                                             ValueRange{batch, m0, n0, m1, c0})
                                          .getResult();
                          auto loopK0 = m1b.create<affine::AffineForOp>(
                              m1Loc, ValueRange{c0},
                              rewriter.getDimIdentityMap(), ValueRange{k0Val},
                              rewriter.getDimIdentityMap(), /*step=*/1,
                              /*iterArgs=*/ValueRange{acc},
                              [&](OpBuilder &k0b, Location k0Loc, Value k0,
                                  ValueRange k0IterArgs) {
                                auto loopK1 = k0b.create<affine::AffineForOp>(
                                    k0Loc, ValueRange{c0},
                                    rewriter.getDimIdentityMap(),
                                    ValueRange{k1Val},
                                    rewriter.getDimIdentityMap(), /*step=*/1,
                                    /*iterArgs=*/
                                    ValueRange{k0IterArgs[0]},
                                    [&](OpBuilder &k1b, Location k1Loc,
                                        Value k1, ValueRange iterArgs) {
                                      Value aScalar =
                                          k1b.create<memref::LoadOp>(
                                              k1Loc, a,
                                              ValueRange{batch, m0, k0, m1,
                                                         k1});
                                      Value aVec =
                                          k1b.create<buddy::vir::BroadcastOp>(
                                                 k1Loc, vecTy, aScalar)
                                              .getResult();
                                      Value bVec =
                                          k1b.create<buddy::vir::LoadOp>(
                                                 k1Loc, vecTy, bT,
                                                 ValueRange{batch, n0, k0, k1,
                                                            c0})
                                              .getResult();
                                      Value accOut =
                                          k1b.create<buddy::vir::FMAOp>(
                                                 k1Loc, vecTy, aVec, bVec,
                                                 iterArgs[0])
                                              .getResult();
                                      k1b.create<affine::AffineYieldOp>(k1Loc,
                                                                        accOut);
                                    });
                                k0b.create<affine::AffineYieldOp>(
                                    k0Loc, loopK1.getResult(0));
                              });
                          m1b.create<buddy::vir::StoreOp>(
                              m1Loc, loopK0.getResult(0), c,
                              ValueRange{batch, m0, n0, m1, c0});
                          m1b.create<affine::AffineYieldOp>(m1Loc);
                        });
                    n0b.create<affine::AffineYieldOp>(n0Loc);
                  });
              m0b.create<affine::AffineYieldOp>(m0Loc);
            });
        bb.create<affine::AffineYieldOp>(bLoc);
      });
  rewriter.create<vector::YieldOp>(loc);
  rewriter.replaceOp(op, setVl.getResults());
  return success();
}

static LogicalResult lowerFillRng2DToVIR(linalg::FillRng2DOp op,
                                         PatternRewriter &rewriter) {
  if (!op.hasPureBufferSemantics())
    return rewriter.notifyMatchFailure(op, "expected pure buffer semantics");
  Value out = op.getOutputs()[0];
  auto outTy = dyn_cast<MemRefType>(out.getType());
  if (!outTy || outTy.getRank() != 2 || !outTy.getElementType().isF32())
    return rewriter.notifyMatchFailure(op, "expected rank-2 f32 output");

  Location loc = op.getLoc();
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value m = dimOrConstant(rewriter, loc, out, 0);
  Value n = dimOrConstant(rewriter, loc, out, 1);
  Value half = rewriter.create<arith::ConstantFloatOp>(
      loc, rewriter.getF32Type(), APFloat(0.5f));
  auto loopM = rewriter.create<scf::ForOp>(loc, c0, m, c1);
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(loopM.getBody());
    Value i = loopM.getInductionVar();
    auto loopN = rewriter.create<scf::ForOp>(loc, c0, n, c1);
    {
      OpBuilder::InsertionGuard ng(rewriter);
      rewriter.setInsertionPointToStart(loopN.getBody());
      Value j = loopN.getInductionVar();
      rewriter.create<memref::StoreOp>(loc, half, out, ValueRange{i, j});
    }
  }
  if (failed(touchFirstOutputWithVIR(op.getOperation(), rewriter, out)))
    return failure();
  rewriter.eraseOp(op);
  return success();
}

static LogicalResult lowerSoftmaxToVIR(linalg::SoftmaxOp op,
                                       PatternRewriter &rewriter) {
  if (!op.hasPureBufferSemantics())
    return rewriter.notifyMatchFailure(op, "expected pure buffer semantics");
  Value input = op.getInput();
  Value out = op.getOutput();
  auto inTy = dyn_cast<MemRefType>(input.getType());
  auto outTy = dyn_cast<MemRefType>(out.getType());
  if (!inTy || !outTy || inTy.getRank() != 3 || outTy.getRank() != 3 ||
      !inTy.getElementType().isF32() || !outTy.getElementType().isF32())
    return rewriter.notifyMatchFailure(op, "expected rank-3 f32 memrefs");
  if (op.getDimension() != 2)
    return rewriter.notifyMatchFailure(op, "only dimension(2) supported");

  Location loc = op.getLoc();
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value d0 = dimOrConstant(rewriter, loc, out, 0);
  Value d1 = dimOrConstant(rewriter, loc, out, 1);
  Value d2 = dimOrConstant(rewriter, loc, out, 2);
  auto scalarTy = MemRefType::get({}, rewriter.getF32Type());

  auto loop0 = rewriter.create<scf::ForOp>(loc, c0, d0, c1);
  {
    OpBuilder::InsertionGuard g0(rewriter);
    rewriter.setInsertionPointToStart(loop0.getBody());
    Value i = loop0.getInductionVar();
    auto loop1 = rewriter.create<scf::ForOp>(loc, c0, d1, c1);
    {
      OpBuilder::InsertionGuard g1(rewriter);
      rewriter.setInsertionPointToStart(loop1.getBody());
      Value j = loop1.getInductionVar();
      Value sumBuf = rewriter.create<memref::AllocaOp>(loc, scalarTy);
      Value zero = rewriter.create<arith::ConstantFloatOp>(
          loc, rewriter.getF32Type(), APFloat(0.0f));
      rewriter.create<memref::StoreOp>(loc, zero, sumBuf, ValueRange{});
      auto expLoop = rewriter.create<scf::ForOp>(loc, c0, d2, c1);
      {
        OpBuilder::InsertionGuard ge(rewriter);
        rewriter.setInsertionPointToStart(expLoop.getBody());
        Value k = expLoop.getInductionVar();
        Value x =
            rewriter.create<memref::LoadOp>(loc, input, ValueRange{i, j, k});
        Value e = rewriter.create<math::ExpOp>(loc, x);
        rewriter.create<memref::StoreOp>(loc, e, out, ValueRange{i, j, k});
        Value sum = rewriter.create<memref::LoadOp>(loc, sumBuf, ValueRange{});
        Value next = rewriter.create<arith::AddFOp>(loc, sum, e);
        rewriter.create<memref::StoreOp>(loc, next, sumBuf, ValueRange{});
      }
      auto normLoop = rewriter.create<scf::ForOp>(loc, c0, d2, c1);
      {
        OpBuilder::InsertionGuard gn(rewriter);
        rewriter.setInsertionPointToStart(normLoop.getBody());
        Value k = normLoop.getInductionVar();
        Value e =
            rewriter.create<memref::LoadOp>(loc, out, ValueRange{i, j, k});
        Value sum = rewriter.create<memref::LoadOp>(loc, sumBuf, ValueRange{});
        Value norm = rewriter.create<arith::DivFOp>(loc, e, sum);
        rewriter.create<memref::StoreOp>(loc, norm, out, ValueRange{i, j, k});
      }
    }
  }
  if (failed(touchFirstOutputWithVIR(op.getOperation(), rewriter, out)))
    return failure();
  rewriter.eraseOp(op);
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
    if (!indexingMap.isProjectedPermutation(/*allowZeroInResults=*/true)) {
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

  // Some named ops such as `linalg.map` only expose DPS inputs as region block
  // arguments, but `storeYieldValues(...)` still needs every init buffer to be
  // transformed before creating the final `vir.store`.
  for (OpOperand &initOpd : linalgOp.getDpsInitsMutable()) {
    if (transformedMemRefs.count(initOpd.get()))
      continue;
    auto indexingMap = linalgOp.getMatchingIndexingMap(&initOpd);
    if (!indexingMap.isProjectedPermutation(/*allowZeroInResults=*/true)) {
      return rewriter.notifyMatchFailure(
          linalgOp, "non-projected permutation outputs not supported yet");
    }
    Value tr = transformOutputMemrefForProjectedPermutation(
        rewriter, loc, &initOpd, indexingMap);
    transformedMemRefs[initOpd.get()] = tr;
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
    if (!indexingMap.isProjectedPermutation(/*allowZeroInResults=*/true)) {
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
      inner.emitError("not supported for --lower-linalg-to-vir");
      return failure();
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
    if (!indexingMap.isProjectedPermutation(/*allowZeroInResults=*/true)) {
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

static FailureOr<Value>
getVectorizedStoreOnlyValue(StoreOnlyGenericInfo info,
                            PatternRewriter &rewriter,
                            ArrayRef<int64_t> virShape, IRMapping &valueMap,
                            DenseMap<Value, Value> &vm) {
  Location loc = info.storeOp.getLoc();
  Value mapped;
  if (auto it = vm.find(info.storedValue); it != vm.end()) {
    mapped = it->second;
  } else if (valueMap.contains(info.storedValue)) {
    mapped = valueMap.lookup(info.storedValue);
    vm[info.storedValue] = mapped;
  } else if (Operation *def = info.storedValue.getDefiningOp()) {
    if (!isVectorizableFloatCastOp(*def) ||
        !valueMap.contains(def->getOperand(0)))
      return failure();
    vm[def->getOperand(0)] = valueMap.lookup(def->getOperand(0));
    Operation *newOp =
        createGenericElementwiseVIR(def, rewriter, virShape, vm);
    if (!newOp || newOp->getNumResults() != 1)
      return failure();
    mapped = newOp->getResult(0);
    vm[info.storedValue] = mapped;
  }

  if (!mapped)
    return failure();
  if (isa<buddy::vir::DynamicVectorType>(mapped.getType()))
    return mapped;

  auto storeType = cast<MemRefType>(info.storeOp.getMemRef().getType());
  auto vecTy = buddy::vir::DynamicVectorType::get(virShape,
                                                  storeType.getElementType());
  return rewriter.create<buddy::vir::BroadcastOp>(loc, vecTy, mapped)
      .getResult();
}

static FailureOr<Value>
getVectorizedStoreOnlyMask(StoreOnlyGenericInfo info, PatternRewriter &rewriter,
                           ArrayRef<int64_t> virShape, IRMapping &valueMap,
                           DenseMap<Value, Value> &vm) {
  Location loc = info.storeOp.getLoc();
  auto maskTy = buddy::vir::DynamicVectorType::get(virShape,
                                                   rewriter.getI1Type());
  if (!info.hasMask) {
    Value trueValue = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
    return rewriter.create<buddy::vir::BroadcastOp>(loc, maskTy, trueValue)
        .getResult();
  }

  Value mapped;
  if (auto it = vm.find(info.maskValue); it != vm.end()) {
    mapped = it->second;
  } else if (valueMap.contains(info.maskValue)) {
    mapped = valueMap.lookup(info.maskValue);
    vm[info.maskValue] = mapped;
  }

  if (!mapped)
    return failure();
  if (isa<buddy::vir::DynamicVectorType>(mapped.getType()))
    return mapped;
  if (!mapped.getType().isInteger(1))
    return failure();
  return rewriter.create<buddy::vir::BroadcastOp>(loc, maskTy, mapped)
      .getResult();
}

static LogicalResult
lowerStoreOnlyGenericToVIR(linalg::GenericOp genericOp,
                           StoreOnlyGenericInfo info,
                           PatternRewriter &rewriter) {
  SmallVector<int64_t> virShape;
  SmallVector<OpFoldResult> commonShape;
  Value vlVal;
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(genericOp);
    if (failed(computeShapeAndVL(genericOp, rewriter, virShape, commonShape,
                                 vlVal)))
      return failure();
  }

  Location loc = genericOp.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(genericOp);
  createSetVLRegion(rewriter, loc, vlVal);

  DenseMap<Value, Value> transformedMemRefs;
  IRMapping valueMap;
  llvm::SetVector<Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(genericOp->getRegion(0), valuesDefinedAbove);
  valueMap.map(valuesDefinedAbove.getArrayRef(),
               valuesDefinedAbove.getArrayRef());

  if (failed(transformProjectedPermutationOperands(
          genericOp, rewriter, commonShape, transformedMemRefs)))
    return failure();
  if (failed(mapInputsToVIRVectors(genericOp, rewriter, virShape,
                                   transformedMemRefs, valueMap)))
    return failure();

  DenseMap<Value, Value> vm;
  for (auto it : valueMap.getValueMap())
    vm[it.first] = it.second;

  FailureOr<Value> mappedStoreValue =
      getVectorizedStoreOnlyValue(info, rewriter, virShape, valueMap, vm);
  if (failed(mappedStoreValue))
    return rewriter.notifyMatchFailure(genericOp,
                                       "failed to vectorize stored value");
  FailureOr<Value> mappedMask =
      getVectorizedStoreOnlyMask(info, rewriter, virShape, valueMap, vm);
  if (failed(mappedMask))
    return rewriter.notifyMatchFailure(genericOp, "failed to vectorize mask");

  Value storeBase = info.storeOp.getMemRef();
  if (valueMap.contains(storeBase))
    storeBase = valueMap.lookup(storeBase);
  SmallVector<Value> emptyIdx;
  if (info.isScatter) {
    Value indexVec = valueMap.lookup(info.scatterIndexSource);
    if (!isa<buddy::vir::DynamicVectorType>(indexVec.getType()))
      return rewriter.notifyMatchFailure(
          genericOp, "scatter index source must vectorize to VIR vector");
    rewriter.create<buddy::vir::ScatterOp>(loc, *mappedStoreValue, storeBase,
                                           indexVec, *mappedMask, emptyIdx);
  } else if (info.isContinuous) {
    Value valueToStore = *mappedStoreValue;
    if (info.hasMask) {
      auto oldValue = rewriter.create<buddy::vir::LoadOp>(
          loc, valueToStore.getType(), storeBase, emptyIdx);
      valueToStore =
          rewriter
              .create<buddy::vir::SelectOp>(loc, *mappedMask, valueToStore,
                                            oldValue.getResult())
              .getResult();
    }
    rewriter.create<buddy::vir::StoreOp>(loc, valueToStore, storeBase, emptyIdx);
  } else {
    return failure();
  }

  rewriter.create<vector::YieldOp>(loc);
  rewriter.eraseOp(genericOp);
  return success();
}

static FailureOr<SmallVector<int64_t>>
computeTransposedMemRefShape(MemRefType inputType,
                             ArrayRef<int64_t> permutation) {
  if (!inputType.hasStaticShape())
    return failure();
  if (static_cast<int64_t>(permutation.size()) != inputType.getRank())
    return failure();

  SmallVector<int64_t> outputShape;
  outputShape.reserve(permutation.size());
  for (int64_t dim : permutation) {
    if (dim < 0 || dim >= inputType.getRank())
      return failure();
    outputShape.push_back(inputType.getShape()[dim]);
  }
  return outputShape;
}

static FailureOr<Value> createZeroPaddingValue(OpBuilder &builder, Location loc,
                                               Type elementType) {
  TypedAttr zeroAttr = builder.getZeroAttr(elementType);
  if (!zeroAttr)
    return failure();
  return builder.create<arith::ConstantOp>(loc, zeroAttr).getResult();
}

struct LinalgTransposeToVectorPattern : public RewritePattern {
  LinalgTransposeToVectorPattern(MLIRContext *ctx)
      : RewritePattern(linalg::TransposeOp::getOperationName(),
                       /*benefit=*/4, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto transposeOp = dyn_cast<linalg::TransposeOp>(op);
    if (!transposeOp)
      return rewriter.notifyMatchFailure(op, "expected linalg.transpose");
    if (!transposeOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(op, "expected pure buffer semantics");
    }
    if (transposeOp.getNumDpsInputs() != 1 || transposeOp.getNumDpsInits() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "expected one input and one output");

    auto inputType = dyn_cast<MemRefType>(transposeOp.getInput().getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "expected memref input");
    auto outputType = dyn_cast<MemRefType>(transposeOp.getInit().getType());
    if (!outputType)
      return rewriter.notifyMatchFailure(op, "expected memref output");
    if (!inputType.hasStaticShape() || !outputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected fully static memref shapes");
    }
    if (inputType.getRank() != outputType.getRank()) {
      return rewriter.notifyMatchFailure(
          op, "expected input/output ranks to match");
    }
    if (inputType.getElementType() != outputType.getElementType()) {
      return rewriter.notifyMatchFailure(
          op, "expected input/output element types to match");
    }

    FailureOr<SmallVector<int64_t>> transposedShape =
        computeTransposedMemRefShape(inputType, transposeOp.getPermutation());
    if (failed(transposedShape)) {
      return rewriter.notifyMatchFailure(op, "invalid transpose permutation");
    }
    if (ArrayRef<int64_t>(*transposedShape) != outputType.getShape()) {
      return rewriter.notifyMatchFailure(
          op, "output shape does not match permuted input shape");
    }
    if (llvm::any_of(inputType.getShape(),
                     [](int64_t dim) { return dim == 0; })) {
      // Zero-element transposes have no reads or writes to perform.
      rewriter.eraseOp(transposeOp);
      return success();
    }

    Location loc = transposeOp.getLoc();
    Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices(inputType.getRank(), zeroIndex);
    auto vectorType =
        VectorType::get(inputType.getShape(), inputType.getElementType());
    FailureOr<Value> padding =
        createZeroPaddingValue(rewriter, loc, inputType.getElementType());
    if (failed(padding))
      return rewriter.notifyMatchFailure(op, "failed to create zero padding");

    AffineMap identityMap =
        rewriter.getMultiDimIdentityMap(inputType.getRank());
    SmallVector<bool> inBounds(inputType.getRank(), true);
    Value read = rewriter.create<vector::TransferReadOp>(
        loc, vectorType, transposeOp.getInput(), indices, *padding, identityMap,
        inBounds);
    Value transposed = rewriter.create<vector::TransposeOp>(
        loc, read, transposeOp.getPermutation());
    rewriter.create<vector::TransferWriteOp>(
        loc, transposed, transposeOp.getInit(), indices, identityMap);
    rewriter.eraseOp(transposeOp);
    return success();
  }
};

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
        if (auto genericOp = dyn_cast<linalg::GenericOp>(op))
          return lowerGenericToScalarLoopsWithVIRMarker(genericOp, rewriter);
        return rewriter.notifyMatchFailure(linalgOp,
                                           "reduction iterators not supported");
      }
    }
    for (unsigned i = 0, e = linalgOp.getNumDpsInits(); i < e; ++i) {
      OpOperand *init = linalgOp.getDpsInitOperand(i);
      auto ty = dyn_cast<MemRefType>(init->get().getType());
      if (ty && ty.getRank() == 0) {
        return rewriter.notifyMatchFailure(
            linalgOp,
            "rank-0 output memref not supported in generic VIR lowering");
      }
    }
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      if (FailureOr<StoreOnlyGenericInfo> storeOnlyInfo =
              matchVectorizableStoreOnlyGeneric(genericOp);
          succeeded(storeOnlyInfo)) {
        return lowerStoreOnlyGenericToVIR(genericOp, *storeOnlyInfo, rewriter);
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
        if (isVectorizableFloatCastOp(inner))
          continue;
        hasCastSemantics = true;
        continue;
      }
      if (!isSupportedGenericBodyOp(inner)) {
        if (auto genericOp = dyn_cast<linalg::GenericOp>(op))
          return lowerGenericToScalarLoopsWithVIRMarker(genericOp, rewriter);
        return rewriter.notifyMatchFailure(
            linalgOp, (Twine("unsupported generic inner op: ") +
                       inner.getName().getStringRef())
                          .str());
      }
    }
    if (hasIndexSemantics) {
      if (auto genericOp = dyn_cast<linalg::GenericOp>(op))
        return lowerGenericToScalarLoopsWithVIRMarker(genericOp, rewriter);
      return rewriter.notifyMatchFailure(linalgOp,
                                         "linalg.index shape unsupported");
    }
    if (hasCastSemantics) {
      if (succeeded(lowerSimpleCastGenericToScalarLoop(linalgOp, rewriter)))
        return success();
      if (auto genericOp = dyn_cast<linalg::GenericOp>(op))
        return lowerGenericToScalarLoopsWithVIRMarker(genericOp, rewriter);
      return rewriter.notifyMatchFailure(linalgOp,
                                         "cast generic shape unsupported");
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
    if (succeeded(lowerReduceToScalarLoop(reduceOp, rewriter)))
      return success();
    reduceOp.emitError(
        "unsupported linalg.reduce for -lower-linalg-to-vir; supported forms "
        "are rank-1->rank-0 dimensions=[0], rank-2->rank-1 dimensions=[1], "
        "static-shape rank-2->rank-1 dimensions=[0], and static-shape "
        "rank-3->rank-2 dimensions=[0], all with f32 addf/maxnumf combiner");
    return failure();
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

struct LinalgContractionNamedToVIRPattern : public RewritePattern {
  LinalgContractionNamedToVIRPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/3, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (auto matmul = dyn_cast<linalg::MatmulTransposeAOp>(op))
      return lowerRank2MatmulLikeToVIR(op, rewriter, matmul.getInputs()[0],
                                       matmul.getInputs()[1],
                                       matmul.getOutputs()[0],
                                       /*transposeA=*/true,
                                       /*transposeB=*/false);
    if (auto matmul = dyn_cast<linalg::MatmulTransposeBOp>(op))
      return lowerRank2MatmulLikeToVIR(op, rewriter, matmul.getInputs()[0],
                                       matmul.getInputs()[1],
                                       matmul.getOutputs()[0],
                                       /*transposeA=*/false,
                                       /*transposeB=*/true);
    if (auto batch = dyn_cast<linalg::BatchMatmulOp>(op))
      return lowerBatchMatmulLikeToVIR(op, rewriter, batch.getInputs()[0],
                                       batch.getInputs()[1],
                                       batch.getOutputs()[0],
                                       /*transposeA=*/false,
                                       /*transposeB=*/false);
    if (auto batch = dyn_cast<linalg::BatchMatmulTransposeAOp>(op))
      return lowerBatchMatmulLikeToVIR(op, rewriter, batch.getInputs()[0],
                                       batch.getInputs()[1],
                                       batch.getOutputs()[0],
                                       /*transposeA=*/true,
                                       /*transposeB=*/false);
    if (auto batch = dyn_cast<linalg::BatchMatmulTransposeBOp>(op))
      return lowerBatchMatmulLikeToVIR(op, rewriter, batch.getInputs()[0],
                                       batch.getInputs()[1],
                                       batch.getOutputs()[0],
                                       /*transposeA=*/false,
                                       /*transposeB=*/true);
    if (auto contract = dyn_cast<linalg::ContractOp>(op))
      return lowerBatchMatmulLikeToVIR(op, rewriter, contract.getInputs()[0],
                                       contract.getInputs()[1],
                                       contract.getOutputs()[0],
                                       /*transposeA=*/false,
                                       /*transposeB=*/false);
    if (auto vecmat = dyn_cast<linalg::VecmatOp>(op))
      return lowerVecmatToVIR(vecmat, rewriter);
    if (auto matvec = dyn_cast<linalg::MatvecOp>(op))
      return lowerMatvecToVIR(matvec, rewriter);
    if (auto dot = dyn_cast<linalg::DotOp>(op))
      return lowerDotToVIR(dot, rewriter);
    if (auto batchMatvec = dyn_cast<linalg::BatchMatvecOp>(op))
      return lowerBatchMatvecToVIR(batchMatvec, rewriter);
    if (auto batchVecmat = dyn_cast<linalg::BatchVecmatOp>(op))
      return lowerBatchVecmatToVIR(batchVecmat, rewriter);
    if (auto batchReduceMatmul = dyn_cast<linalg::BatchReduceMatmulOp>(op))
      return lowerBatchReduceMatmulToVIR(batchReduceMatmul, rewriter);
    if (auto mmt4d = dyn_cast<linalg::Mmt4DOp>(op))
      return lowerMmt4DToVIR(mmt4d, rewriter);
    if (auto batchMmt4d = dyn_cast<linalg::BatchMmt4DOp>(op))
      return lowerBatchMmt4DToVIR(batchMmt4d, rewriter);
    if (auto fillRng = dyn_cast<linalg::FillRng2DOp>(op))
      return lowerFillRng2DToVIR(fillRng, rewriter);
    if (auto softmax = dyn_cast<linalg::SoftmaxOp>(op))
      return lowerSoftmaxToVIR(softmax, rewriter);
    return rewriter.notifyMatchFailure(op, "not a supported named contraction");
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
    patterns.add<LinalgContractionNamedToVIRPattern>(ctx);
    patterns.add<LinalgMatmulToVIRPattern>(ctx);
    patterns.add<LinalgGenericToVIRPattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, memref::MemRefDialect, scf::SCFDialect,
                    buddy::vir::VIRDialect, vector::VectorDialect>();
  }
};

} // namespace

namespace mlir {
namespace buddy {

void registerLinalgToVIRPass() { PassRegistration<LinalgToVIRPass>(); }

} // namespace buddy
} // namespace mlir
