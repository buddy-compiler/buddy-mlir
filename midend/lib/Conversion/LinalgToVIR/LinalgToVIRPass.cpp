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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
                                             Location loc, Value vlVal) {
  auto setVl = rewriter.create<buddy::vir::SetVLOp>(
      loc, /*results=*/TypeRange{}, /*operands=*/ValueRange{vlVal});
  Region &region = setVl.getRegion();
  Block &block = region.emplaceBlock();
  rewriter.setInsertionPointToStart(&block);
  return setVl;
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
    if (!mapped && valueMap.contains(yv)) {
      Value scalar = valueMap.lookup(yv);
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
      mapped = rewriter.create<buddy::vir::BroadcastOp>(loc, outVecTy, scalar);
    }
    auto initOpd = linalgOp.getDpsInitOperand(i);
    auto indexingMap = linalgOp.getMatchingIndexingMap(initOpd);
    if (!indexingMap.isProjectedPermutation()) {
      // TODO: Support non-projected-permutation with scatter.
      return rewriter.notifyMatchFailure(
          linalgOp, "non-projected permutation outputs not supported yet");
    }
    Value base = memMap.lookup(initOpd->get());
    SmallVector<Value> zeroIdx(linalgOp.getNumLoops());
    for (int d = 0, e = linalgOp.getNumLoops(); d < e; ++d)
      zeroIdx[d] = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    rewriter.create<buddy::vir::StoreOp>(loc, mapped, base, zeroIdx);
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
    patterns.add<LinalgGenericToVIRPattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, buddy::vir::VIRDialect,
                    vector::VectorDialect>();
  }
};

} // namespace

namespace mlir {
namespace buddy {

void registerLinalgToVIRPass() { PassRegistration<LinalgToVIRPass>(); }

} // namespace buddy
} // namespace mlir
