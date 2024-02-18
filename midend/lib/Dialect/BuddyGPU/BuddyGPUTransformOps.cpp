// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "BuddyGPU/BuddyGPUTransformOps.h"

#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::buddy::buddygpu;

transform_dialect::
    BuddyGPUTransformExtensions::BuddyGPUTransformExtensions() {
  // CreateAsyncGroupsOp depends on the following two dialects.
  declareGeneratedDialect<mlir::gpu::GPUDialect>();
  declareGeneratedDialect<mlir::nvgpu::NVGPUDialect>();

  registerTransformOps<
#define GET_OP_LIST
#include "BuddyGPU/BuddyGPUTransformOps.cpp.inc"
      >();
}

void buddy::registerBuddyTransformOps(DialectRegistry &registry) {
  registry.addExtensions<transform_dialect::BuddyGPUTransformExtensions>();
}

//===---------------------------------------------------------------------===//
// VectorToWarpExecuteOnLane0Op.
//===---------------------------------------------------------------------===//
void transform_dialect::VectorToWarpExecuteOnLane0Op::build(
    OpBuilder &builder, OperationState &result, Value target,
    int64_t warpSize) {
  MLIRContext *ctx = builder.getContext();
  result.addOperands(target);
  result.addAttribute(
      VectorToWarpExecuteOnLane0Op::getWarpSizeAttrName(result.name),
      builder.getI64IntegerAttr(warpSize));
  result.addTypes({transform::AnyOpType::get(ctx)});
}

/// Helper method to replace all uses of the laneId operand by the constant
/// 0 inside the region. This is a necessary prerequisite to perform any kind of
/// hoisting of IR that is inside the region.
/// Return success if any replacement occurred, failure otherwise.
// TODO: this is currently brittle, what we really need here is a scope-aware
// SCCP.
static LogicalResult
replaceAllUsesOfLaneWithin(RewriterBase &b,
                           vector::WarpExecuteOnLane0Op executeOp) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(executeOp);
  Value zero = b.create<arith::ConstantIndexOp>(executeOp.getLoc(), 0);
  b.setInsertionPointToStart(&executeOp.getWarpRegion().front());
  Value laneId = executeOp.getLaneid();
  bool applied = false;
  for (Operation *user : llvm::make_early_inc_range(laneId.getUsers())) {
    if (!executeOp->isProperAncestor(user))
      continue;
    b.startRootUpdate(user);
    user->replaceUsesOfWith(laneId, zero);
    b.finalizeRootUpdate(user);
    applied = true;
  }
  return success(applied);
}

/// Return the gpu::ThreadIdOp for which the predicate if equivalent to
/// `if (threadIdx.x == 0)`.
// TODO: Figure out the proper canonicalization and drop the complexity here.
// TODO: More sophisticated detection for matching
//   (threadIdx.x == 0 && other stuff not involving threadIdx.x)
static FailureOr<gpu::ThreadIdOp> isThreadIdxxZeroPredicate(scf::IfOp ifOp) {
  if (!ifOp || ifOp.getNumResults() > 0 ||
      ifOp.getThenRegion().getBlocks().size() != 1 ||
      !ifOp.getElseRegion().empty())
    return failure();
  auto pred = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!pred)
    return failure();
  auto EQ = arith::CmpIPredicate::eq;
  auto SLT = arith::CmpIPredicate::slt;
  auto SLE = arith::CmpIPredicate::sle;
  auto ULT = arith::CmpIPredicate::ult;
  auto ULE = arith::CmpIPredicate::ule;
  if (auto threadIdOp = pred.getLhs().getDefiningOp<gpu::ThreadIdOp>()) {
    if (threadIdOp.getDimension() != gpu::Dimension::x)
      return failure();
    if (pred.getPredicate() == EQ && isConstantIntValue(pred.getRhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == SLE && isConstantIntValue(pred.getRhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == ULE && isConstantIntValue(pred.getRhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == SLT && isConstantIntValue(pred.getRhs(), 1))
      return threadIdOp;
    if (pred.getPredicate() == ULT && isConstantIntValue(pred.getRhs(), 1))
      return threadIdOp;
  }
  auto SGT = arith::CmpIPredicate::sgt;
  auto SGE = arith::CmpIPredicate::sge;
  auto UGT = arith::CmpIPredicate::ugt;
  auto UGE = arith::CmpIPredicate::uge;
  if (auto threadIdOp = pred.getRhs().getDefiningOp<gpu::ThreadIdOp>()) {
    if (threadIdOp.getDimension() != gpu::Dimension::x)
      return failure();
    if (pred.getPredicate() == EQ && isConstantIntValue(pred.getLhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == SGE && isConstantIntValue(pred.getLhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == UGE && isConstantIntValue(pred.getLhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == SGT && isConstantIntValue(pred.getLhs(), 1))
      return threadIdOp;
    if (pred.getPredicate() == UGT && isConstantIntValue(pred.getLhs(), 1))
      return threadIdOp;
  }
  return failure();
}

struct VectorDistributionResult {
  vector::WarpExecuteOnLane0Op warpOp;
};

static FailureOr<VectorDistributionResult>
rewriteScfIfAsWarpExecuteOnLane0(RewriterBase &rewriter, Location loc,
                                 scf::IfOp ifOp, int64_t workgroupSizeX,
                                 int64_t warpSize) {
  // Bail if cond is not `if (threadIdx.x == 0)`.
  FailureOr<gpu::ThreadIdOp> maybeThreadIdxxOp =
      isThreadIdxxZeroPredicate(ifOp);
  if (failed(maybeThreadIdxxOp))
    return failure();

  // All the code below will be executed on a single warp given a
  // fixed (threadIdxy, threadIdxz). Note, we reuse
  // `maybeThreadIdxxOp` here because we later want to replace this
  // op instance by 0 without relying on CSE or canonicalizations.
  Value threadIdxx = *maybeThreadIdxxOp;

  assert(workgroupSizeX % warpSize == 0);
  if (workgroupSizeX != warpSize) {
    // Add a guard for `threadIdxx < warp size` around the
    // WarpExecuteOnLane0Op.
    Value predicate = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, threadIdxx,
        rewriter.create<arith::ConstantIndexOp>(loc, warpSize));
    // Note: return-less IfOp is built with a terminator, no need to
    // add one.
    auto newIfOp =
        rewriter.create<scf::IfOp>(loc, predicate, /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&newIfOp.getThenRegion().front());
  }
  auto warpOp = rewriter.create<vector::WarpExecuteOnLane0Op>(
      loc, TypeRange(), threadIdxx, warpSize);

  // Move the code from the previous ifOp to the
  // WarpExecuteOnLane0Op.
  Block &sourceBlock = ifOp.getThenRegion().front();
  Block &targetBlock = warpOp.getWarpRegion().front();
  Block::iterator insertionPoint = targetBlock.begin();
  targetBlock.getOperations().splice(insertionPoint,
                                     sourceBlock.getOperations(),
                                     sourceBlock.without_terminator().begin(),
                                     sourceBlock.without_terminator().end());
  rewriter.setInsertionPointToEnd(&targetBlock);
  rewriter.create<vector::YieldOp>(loc);

  // Erase old op.
  rewriter.eraseOp(ifOp);

  // This simple rewrite propagates zero in lieu of laneId within the
  // warp_execute_on_lane_0 op.
  // Atm, this **must** occur before any hoisting of code.
  // TODO: Replace this by a more robust scoped SCCP that will make
  // it more robust re. hoisting.
  (void)replaceAllUsesOfLaneWithin(rewriter, warpOp);

  // Hoist the scalar code outside of the warp region.
  // Note: moving code does not require a listener.
  vector::moveScalarUniformCode(warpOp);

  return VectorDistributionResult{warpOp};
}

// TODO: Refactor in a generic util that can be reused.
// static HAL::ExecutableExportOp
// getExecutableExportOpForFunc(HAL::ExecutableVariantOp halExecutableVariantOp,
//                              func::FuncOp funcOp) {
//   if (!halExecutableVariantOp || !funcOp)
//     return {};
//   HAL::ExecutableExportOp exportOp;
//   halExecutableVariantOp->walk([&](HAL::ExecutableExportOp op) {
//     if (op.getSymName() != funcOp.getName())
//       return WalkResult::advance();
//     exportOp = op;
//     return WalkResult::interrupt();
//   });
//   return exportOp;
// }

DiagnosedSilenceableFailure
transform_dialect::VectorToWarpExecuteOnLane0Op::applyToOne(
    transform::TransformRewriter &rewriter, scf::IfOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  // if (!isa<HAL::ExecutableOp, HAL::ExecutableVariantOp>(state.getTopLevel())) {
  //   results.assign(1, nullptr);
  //   return emitDefaultSilenceableFailure(state.getTopLevel())
  //          << "requires HAL::ExecutableOp or "
  //             "HAL::ExecutableVariantOp toplevel "
  //             "so that IR is properly isolated. This is required so "
  //             "we can "
  //             "safely inspect the HAL::ExecutableExportOp under "
  //             "multi-threaded "
  //             "pass assumptions.";
  // }

  // auto halExecutableVariantOp =
  //     target->getParentOfType<HAL::ExecutableVariantOp>();
  // auto funcOp = target->getParentOfType<func::FuncOp>();
  // HAL::ExecutableExportOp exportOp =
  //     getExecutableExportOpForFunc(halExecutableVariantOp, funcOp);
  // if (!halExecutableVariantOp || !funcOp || !exportOp) {
  //   // Return a silenceable failure and set the expected 1 result to
  //   // nullptr.
  //   results.assign(1, nullptr);
  //   return emitDefaultSilenceableFailure(target)
  //          << "export op is missing --- the transform is not "
  //             "applied";
  // }

  //std::optional<ArrayAttr> maybeAttr = exportOp.getWorkgroupSize();
  // TODO: Pervasive 3 constant in IREE.
  // if (!maybeAttr || maybeAttr->size() != 3) {
  //   // Return a silenceable failure and set the expected 1 result to
  //   // nullptr.
  //   results.assign(1, nullptr);
  //   return emitDefaultSilenceableFailure(target)
  //          << "export op must have workgroup_size attribute set "
  //             "with 3 entries "
  //             "--- the transform is not applied";
  // }

  // int64_t workgroupSizeX = llvm::cast<IntegerAttr>((*maybeAttr)[0]).getInt();
  // int64_t warpSize = getWarpSize();
  // if (workgroupSizeX % warpSize != 0) {
  //   // Return a silenceable failure and set the expected 1 result to
  //   // nullptr.
  //   results.assign(1, nullptr);
  //   return emitDefaultSilenceableFailure(target)
  //          << "vector distribution requires workgroup size for x to "
  //             "be a "
  //          << "multiple of the warp size: " << workgroupSizeX << " vs "
  //          << warpSize << " --- the transform is not applied";
  // }

  // Location loc = target->getLoc();
  // rewriter.setInsertionPoint(target);
  // FailureOr<VectorDistributionResult> vectorDistributionResult =
  //     rewriteScfIfAsWarpExecuteOnLane0(rewriter, loc, target, workgroupSizeX,
  //                                      warpSize);
  // if (failed(vectorDistributionResult)) {
  //   // Return a silenceable failure and set the expected 1 result to
  //   // nullptr.
  //   results.assign(1, nullptr);
  //   return mlir::emitSilenceableFailure(
  //       target, "scf::ifOp needs to be predicated on threadIdx.x == 0 "
  //               "--- the transform is not applied");
  // }

  // results.push_back(vectorDistributionResult->warpOp);
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// VectorWarpDistributionOp.
//===---------------------------------------------------------------------===//
void transform_dialect::VectorWarpDistributionOp::build(OpBuilder &builder,
                                                        OperationState &result,
                                                        Value target) {
  result.addOperands(target);
}

void transform_dialect::VectorWarpDistributionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

/// Emit shared local memory allocation in case it is needed when lowering the
/// warp operations.
static Value allocateGlobalSharedMemory(Location loc, OpBuilder &builder,
                                        vector::WarpExecuteOnLane0Op warpOp,
                                        Type type) {
  MemRefType memrefType;
  auto addressSpaceAttr = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  if (auto vectorType = llvm::dyn_cast<VectorType>(type)) {
    memrefType =
        MemRefType::get(vectorType.getShape(), vectorType.getElementType(),
                        MemRefLayoutAttrInterface{}, addressSpaceAttr);
  } else {
    memrefType = MemRefType::get({1}, type, MemRefLayoutAttrInterface{},
                                 addressSpaceAttr);
  }
  return builder.create<memref::AllocOp>(loc, memrefType);
}

/// Return a value yielded by `warpOp` which statifies the filter lamdba
/// condition and is not dead.
static OpOperand *getWarpResult(vector::WarpExecuteOnLane0Op warpOp,
                                function_ref<bool(Operation *)> fn) {
  auto yield = cast<vector::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
  for (OpOperand &yieldOperand : yield->getOpOperands()) {
    Value yieldValues = yieldOperand.get();
    Operation *definedOp = yieldValues.getDefiningOp();
    if (definedOp && fn(definedOp)) {
      if (!warpOp.getResult(yieldOperand.getOperandNumber()).use_empty())
        return &yieldOperand;
    }
  }
  return {};
}

namespace {
/// Pattern to convert InsertElement to broadcast, this is a workaround
/// until MultiDimReduction distribution is supported.
class InsertElementToBroadcast final
    : public OpRewritePattern<vector::InsertElementOp> {
public:
  using OpRewritePattern<vector::InsertElementOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertElementOp insertOp,
                                PatternRewriter &rewriter) const override {
    if (insertOp.getDestVectorType().getNumElements() != 1)
      return failure();
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        insertOp, insertOp.getDestVectorType(), insertOp.getSource());
    return success();
  }
};

/// Sink out load op feeding into a warp op yield.
/// ```
/// %0 = vector.warp_execute_on_lane_0(%arg0) -> (f32) {
///   ...
//    %2 = memref.load %src[%c0] : memref<1024xf32>
///   vector.yield %2 : f32
/// }
/// ```
/// To
/// ```
/// %dead = vector.warp_execute_on_lane_0(%arg0) -> (f32) {
///   ...
//    %2 = memref.load %src[%c0] : memref<1024xf32>
///   vector.yield %2 : f32
/// }
/// gpu.synchronize
/// %0 = memref.load %src[%c0] : memref<1024xf32>
struct WarpOpLoad : public OpRewritePattern<vector::WarpExecuteOnLane0Op> {
  using OpRewritePattern<vector::WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand = getWarpResult(
        warpOp, [](Operation *op) { return isa<memref::LoadOp>(op); });
    if (!operand)
      return failure();
    auto load = operand->get().getDefiningOp<memref::LoadOp>();
    unsigned operandIndex = operand->getOperandNumber();
    Value distributedVal = warpOp.getResult(operandIndex);

    SmallVector<Value> indices(load.getIndices().begin(),
                               load.getIndices().end());
    if (!indices.empty())
      return failure();

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(warpOp);
    // TODO: generalize this.
    // options.warpSyncronizationFn currently must take a
    // WarpExecuteOnLane0Op which we don't have here.
    rewriter.create<gpu::BarrierOp>(load.getLoc());
    Value newRead = rewriter.create<memref::LoadOp>(
        load.getLoc(), distributedVal.getType(), load.getMemref(), indices);

    // The result type of WarpExecuteOnLane0Op may or may not match
    // the yielded type depending on whether the op has "broadcast"
    // behavior (see the doc of WarpExecuteOnLane0Op).
    for (OpOperand &use : distributedVal.getUses()) {
      rewriter.startRootUpdate(use.getOwner());
      Value replacement = newRead;
      if (use.get().getType() != newRead.getType()) {
        replacement = rewriter.create<vector::BroadcastOp>(
            load.getLoc(), use.get().getType(), newRead);
      }
      use.getOwner()->setOperand(use.getOperandNumber(), replacement);
      rewriter.finalizeRootUpdate(use.getOwner());
    }
    return success();
  }
};

/// Shared memory allocations are representated as AllocOp in IREE but they
/// really have the semantic of global variables. Therefore hoisting them is
/// always correct for static allocations.
struct HoistSharedMemoryAlloc : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                                PatternRewriter &rewriter) const override {
    // if (!iree_compiler::hasSharedMemoryAddressSpace(alloc.getType()))
    //   return failure();
    auto warpParent = alloc->getParentOfType<vector::WarpExecuteOnLane0Op>();
    if (!warpParent)
      return failure();
    alloc->moveBefore(warpParent);
    // Conservatively move the dealloc after the warpOp. This may
    // extend the liverange of the allocation but is always correct.
    for (Operation *user : alloc->getUsers()) {
      if (isa<memref::DeallocOp>(user))
        user->moveAfter(warpParent);
    }
    return success();
  }
};

} // namespace

static void populateMultiReductionLoweringPatterns(Operation *target,
                                                   RewritePatternSet &patterns,
                                                   PatternBenefit benefit) {
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());

  vector::populateVectorMultiReductionLoweringPatterns(
      patterns, vector::VectorMultiReductionLowering::InnerReduction, benefit);
  patterns.add<InsertElementToBroadcast>(target->getContext(), benefit);
}

static AffineMap simpleDistributionFunction(Value val) {
  AffineMap map = AffineMap::get(val.getContext());
  auto vecType = llvm::dyn_cast<VectorType>(val.getType());
  if (!vecType)
    return map;
  // Create a map (d0, d1) -> (d1) to distribute along the inner
  // dimension. Once we support n-d distribution we can add more
  // complex cases.
  int64_t vecRank = vecType.getRank();
  OpBuilder builder(val.getContext());
  map = AffineMap::get(vecRank, 0, builder.getAffineDimExpr(vecRank - 1));
  return map;
}

static void populateVectorTransferWriteDistribution(Operation *target,
                                                    RewritePatternSet &patterns,
                                                    PatternBenefit benefit) {
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());
  vector::populateDistributeTransferWriteOpPatterns(
      patterns, simpleDistributionFunction, benefit);
}

static Value simpleWarpShuffleFunction(Location loc, OpBuilder &builder,
                                       Value val, Value srcIdx,
                                       int64_t warpSz) {
  assert((val.getType().isF32() || val.getType().isInteger(32)) &&
         "unsupported shuffle type");
  Type i32Type = builder.getIntegerType(32);
  Value srcIdxI32 = builder.create<arith::IndexCastOp>(loc, i32Type, srcIdx);
  Value warpSzI32 = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(i32Type, warpSz));
  Value result = builder
                     .create<gpu::ShuffleOp>(loc, val, srcIdxI32, warpSzI32,
                                             gpu::ShuffleMode::IDX)
                     .getResult(0);
  return result;
}

static void populatePropagateVectorDistribution(Operation *target,
                                                RewritePatternSet &patterns,
                                                PatternBenefit benefit,
                                                unsigned subgroupSize) {
  // auto groupReductionFn =
  //     [subgroupSize](Location loc, OpBuilder &builder, Value input,
  //                    vector::CombiningKind kind, uint32_t size) {
  //       return mlir::iree_compiler::emitGPUGroupReduction(
  //           loc, builder, input, kind, size, subgroupSize,
  //           /*expandSubgroupReduce=*/true);
  //     };
  // assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());
  // vector::populatePropagateWarpVectorDistributionPatterns(
  //     patterns, simpleDistributionFunction, simpleWarpShuffleFunction, benefit);
  // vector::populateDistributeReduction(patterns, groupReductionFn, benefit);
  // patterns.add<WarpOpLoad, HoistSharedMemoryAlloc>(target->getContext(),
  //                                                  benefit);
}

static void warpSyncronizationFn(Location loc, OpBuilder &builder,
                                 vector::WarpExecuteOnLane0Op warpOp) {
  builder.create<gpu::BarrierOp>(loc);
};

static void populateWarpExecuteOnLane0ToScf(
    Operation *target, RewritePatternSet &patterns,
    const vector::WarpExecuteOnLane0LoweringOptions &options,
    PatternBenefit benefit) {
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());
  vector::populateWarpExecuteOnLane0OpToScfForPattern(patterns, options,
                                                      benefit);
}

DiagnosedSilenceableFailure
transform_dialect::VectorWarpDistributionOp::applyToOne(
    transform::TransformRewriter &rewriter, func::FuncOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  // FailureOr<IREE::HAL::ExecutableExportOp> maybeExportOp =
  //     getEntryPoint(target);
  // if (failed(maybeExportOp)) {
  //   state.getTopLevel()->emitOpError("no IREE::HAL::ExecutableExportOp found");
  //   return emitDefaultDefiniteFailure(target);
  // }
  // IREE::HAL::ExecutableExportOp exportOp = *maybeExportOp;

  // std::optional<llvm::APInt> subgroupSize = exportOp.getSubgroupSize();
  // if (!subgroupSize) {
  //   state.getTopLevel()->emitOpError(
  //       "could not extract subgroup size from IREE::HAL::ExecutableExportOp");
  //   return emitDefaultDefiniteFailure(target);
  // }

  // // TODO: Hook up into the ApplyPatternOp in CommonExtensions.cpp to
  // // automatically get listening capabilities.

  // MLIRContext *ctx = target->getContext();
  // // MultiReduction lowering is necessary until we have explicit
  // // support for distributing that op.
  // RewritePatternSet preProcessingPatterns(ctx);
  // populateMultiReductionLoweringPatterns(target, preProcessingPatterns,
  //                                        /*benefit=*/1);
  // vector::ShapeCastOp::getCanonicalizationPatterns(preProcessingPatterns, ctx);
  // vector::BroadcastOp::getCanonicalizationPatterns(preProcessingPatterns, ctx);
  // vector::ExtractOp::getCanonicalizationPatterns(preProcessingPatterns, ctx);
  // transform::ErrorCheckingTrackingListener listener(state, *this);
  // auto checkErrors = llvm::make_scope_exit([&]() {
  //   // The TrackingListener API makes checking for errors mandatory. It is safe
  //   // to drop payload ops during this transform, so we can ignore all errors.
  //   (void)listener.checkAndResetError();
  // });
  // GreedyRewriteConfig config;
  // config.listener = &listener;
  // if (failed(applyPatternsAndFoldGreedily(
  //         target, std::move(preProcessingPatterns), config))) {
  //   return mlir::emitDefiniteFailure(target,
  //                                    "multi-reduce patterns failed to apply");
  // }

  // RewritePatternSet patterns(ctx);
  // populateVectorTransferWriteDistribution(target, patterns,
  //                                         /*benefit=*/2);
  // populatePropagateVectorDistribution(target, patterns,
  //                                     /*benefit=*/1,
  //                                     subgroupSize->getSExtValue());
  // if (failed(
  //         applyPatternsAndFoldGreedily(target, std::move(patterns), config))) {
  //   return mlir::emitDefiniteFailure(
  //       target, "warp distribution patterns failed to apply");
  // }

  // RewritePatternSet endPatterns(ctx);
  // vector::WarpExecuteOnLane0LoweringOptions options;
  // options.warpAllocationFn = allocateGlobalSharedMemory;
  // options.warpSyncronizationFn = warpSyncronizationFn;
  // populateWarpExecuteOnLane0ToScf(target, endPatterns, options,
  //                                 /*benefit=*/0);
  // if (failed(applyPatternsAndFoldGreedily(target, std::move(endPatterns),
  //                                         config))) {
  //   return mlir::emitDefiniteFailure(
  //       target, "warp execute on lane 0 to scf patterns failed to apply");
  // }

  return DiagnosedSilenceableFailure::success();
}

#define GET_OP_CLASSES
#include "BuddyGPU/BuddyGPUTransformOps.cpp.inc"
