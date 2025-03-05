//===- TransformOps.cpp ---------------------------------------------------===//
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
// The process in this file references the IREE project,
// which is hereby acknowledged.
// For the license of the IREE project
// please see: https://github.com/iree-org/iree/blob/main/LICENSE
//
//===----------------------------------------------------------------------===//
//
// This file implements transform ops for GPU targets.
//
//===----------------------------------------------------------------------===//

#include "GPU/TransformOps.h"

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
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <iostream>
#include <optional>

#include "Utils/GPUUtils.h"

using namespace mlir;
using namespace mlir::buddy;

using llvm::dbgs;

#define DEBUG_TYPE "transform-llvmgpu-extensions"
#define DEBUG_TYPE_ALIAS "transform-llvmgpu-extensions-alias"
#define DEBUG_VECTOR_TO_MMA "transform-llvmgpu-extensions-vector-to-mma"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(dbgs() << '[' << DEBUG_TYPE << "] " << X)
#define DBGS_ALIAS() (dbgs() << '[' << DEBUG_TYPE_ALIAS << "] ")
#define DBGS_VECTOR_TO_MMA() (dbgs() << '[' << DEBUG_VECTOR_TO_MMA << "] ")

buddy::gpu::TransformExtensions::TransformExtensions() {
  // CreateAsyncGroupsOp depends on the following two dialects.
  declareGeneratedDialect<mlir::gpu::GPUDialect>();
  declareGeneratedDialect<mlir::nvgpu::NVGPUDialect>();

  registerTransformOps<
#define GET_OP_LIST
#include "GPU/TransformOps.cpp.inc"
      >();
}

void buddy::registerBuddyGPUTransformOps(DialectRegistry &registry) {
  registry.addExtensions<buddy::gpu::TransformExtensions>();
}

//===----------------------------------------------------------------------===//
// HoistStaticAllocOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure buddy::gpu::HoistStaticAllocOp::applyToOne(
    mlir::transform::TransformRewriter &rewriter, func::FuncOp target,
    mlir::transform::ApplyToEachResultList &results,
    mlir::transform::TransformState &state) {
  hoistStaticallyBoundAllocationsInFunc<memref::AllocOp>(rewriter, target);
  return DiagnosedSilenceableFailure::success();
}

void buddy::gpu::HoistStaticAllocOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  mlir::transform::onlyReadsHandle(getTarget(), effects);
  mlir::transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// ApplyUnrollVectorsGpuMmaSyncPatternsOp
//===---------------------------------------------------------------------===//

static std::optional<SmallVector<int64_t>>
getGPUTensorCoreNativeMmaSyncVectorSize(Operation *op) {
  return buddy::gpu::getMmaNativeVectorSize(op);
}

void buddy::gpu::ApplyUnrollVectorsGpuMmaSyncPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  auto unrollOrder = [](Operation *op) -> std::optional<SmallVector<int64_t>> {
    auto contract = dyn_cast<vector::ContractionOp>(op);
    if (!contract)
      return std::nullopt;
    return gpuMmaUnrollOrder(contract);
  };
  vector::populateVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShapeFn(getGPUTensorCoreNativeMmaSyncVectorSize)
                    .setUnrollTraversalOrderFn(unrollOrder));
}

//===---------------------------------------------------------------------===//
// VectorToMMAConversionOp
//===---------------------------------------------------------------------===//

void buddy::gpu::VectorToMMAConversionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  mlir::transform::onlyReadsHandle(getTarget(), effects);
  mlir::transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure
buddy::gpu::VectorToMMAConversionOp::applyToOne(
    mlir::transform::TransformRewriter &rewriter, Operation *target,
    mlir::transform::ApplyToEachResultList &results,
    mlir::transform::TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    // target->emitOpError(
    //     "applies only to isolated-from-above targets because it "
    //     "needs to apply "
    //     "patterns greedily");
    // return emitDefaultDefiniteFailure(target);
  }

  auto funcOp = dyn_cast<func::FuncOp>(target);
  if (!funcOp) {
    target->emitOpError("Must apply to a func op");
    return emitDefaultDefiniteFailure(target);
  }

  if (!(getUseMmaSync() ^ getUseWmma())) {
    target->emitOpError(
        "Exactly one of use_mma_sync or use_wmma must be specified");
    return emitDefaultDefiniteFailure(target);
  }

  MLIRContext *ctx = target->getContext();
  mlir::transform::ErrorCheckingTrackingListener listener(state, *this);
  GreedyRewriteConfig config;
  config.listener = &listener;

  // Unrolling to native vector size must have previously occurred.
  // TODO: Add pattern to propagate the extract through the scf.for
  // ops. Convert slice of contract operations to mma_sync/wmma ops.
  RewritePatternSet patterns(ctx);
  mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
  populatePrepareVectorToMMAPatterns(patterns, getUseMmaSync());
  if (failed(
          applyPatternsAndFoldGreedily(target, std::move(patterns), config))) {
    target->emitOpError("vector to mma preparation patterns failed to apply");
    return emitDefaultDefiniteFailure(target);
  }

  auto diag = DiagnosedSilenceableFailure::success();
  if (getUseWmma()) {
    if (failed(convertVectorToMMAOps(rewriter, target)))
      return mlir::emitDefiniteFailure(
          target, "vector to wmma patterns failed to apply");
    return listener.checkAndResetError();
  }

  if (failed(convertVectorToNVVMCompatibleMMASync(rewriter, funcOp)))
    return mlir::emitDefiniteFailure(target,
                                     "vector to mma patterns failed to apply");

  // Using TF32 for Float.
  RewritePatternSet f32ToTF32patterns(funcOp.getContext());
  nvgpu::populateMmaSyncF32ToTF32Patterns(f32ToTF32patterns,
                                          nvgpu::MmaSyncF32Lowering::TF32);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(f32ToTF32patterns),
                                          config)))
    return mlir::emitDefiniteFailure(
        target, "vector to mma F32ToTF32 patterns failed to apply");

  return listener.checkAndResetError();
}

//===---------------------------------------------------------------------===//
// EliminateGpuBarriersOp
//===---------------------------------------------------------------------===//

/// Returns `true` if the op is known not to have any side effects, but does not
/// implement the MemoryEffectsOpInterface in the suitable way.
static bool isKnownNoEffectsOpWithoutInterface(Operation *op) {
  // memref::AssumeAlignment is conceptually pure, but marking it as such would
  // make DCE immediately remove it.
  return isa<memref::AssumeAlignmentOp>(op);
}

/// Returns `true` if the op is defines the parallel region that is subject to
/// barrier synchronization.
static bool isParallelRegionBoundary(Operation *op) {
  if (op->hasAttr("__parallel_region_boundary_for_test"))
    return true;

  // We consider functions inside executable variants that have the same symbol
  // name as an export symbol.
  auto func = dyn_cast<FunctionOpInterface>(op);
  if (!func)
    return false;
  auto parent = op->getParentOfType<ModuleOp>();
  if (!parent)
    return false;
  auto variant = parent->getParentOfType<mlir::gpu::LaunchOp>();
  if (!variant)
    return false;
  return true;
}

/// Returns `true` if the op behaves like a sequential loop, e.g., the control
/// flow "wraps around" from the end of the body region back to its start.
static bool isSequentialLoopLike(Operation *op) { return isa<scf::ForOp>(op); }

/// Returns `true` if the regions of the op are guaranteed to be executed at
/// most once. Thus, if an operation in one of the nested regions of `op` is
/// executed than so are all the other operations in this region.
static bool hasSingleExecutionBody(Operation *op) {
  return isa<scf::IfOp, memref::AllocaScopeOp>(op);
}

/// Returns `true` if the operation is known to produce a pointer-like object
/// distinct from any other object produced by a similar operation. For example,
/// an allocation produces such an object.
static bool producesDistinctBase(Operation *op) {
  return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(op);
}

/// Populates `effects` with all memory effects without associating them to a
/// specific value.
static void addAllValuelessEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Read>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Write>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Allocate>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Free>());
}

/// Collect the memory effects of the given op in 'effects'. Returns 'true' if
/// it could extract the effect information from the op, otherwise returns
/// 'false' and conservatively populates the list with all possible effects
/// associated with no particular value or symbol.
static bool
collectEffects(Operation *op,
               SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
               bool ignoreBarriers = true) {
  // Skip over barriers to avoid infinite recursion (those barriers would ask
  // this barrier again).
  if (ignoreBarriers && isa<mlir::gpu::BarrierOp>(op))
    return true;

  // Skip over ops that we know have no effects.
  if (isKnownNoEffectsOpWithoutInterface(op))
    return true;

  // Collect effect instances the operation. Note that the implementation of
  // getEffects erases all effect instances that have the type other than the
  // template parameter so we collect them first in a local buffer and then
  // copy.
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> localEffects;
    iface.getEffects(localEffects);
    llvm::append_range(effects, localEffects);
    return true;
  }
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &innerOp : block)
          if (!collectEffects(&innerOp, effects, ignoreBarriers))
            return false;
      }
    }
    return true;
  }

  // We need to be conservative here in case the op doesn't have the interface
  // and assume it can have any possible effect.
  addAllValuelessEffects(effects);
  return false;
}

/// Collects memory effects from operations that may be executed before `op` in
/// a trivial structured control flow, e.g., without branches. Stops at the
/// parallel region boundary or at the barrier operation if `stopAtBarrier` is
/// set. Returns `true` if the memory effects added to `effects` are exact,
/// `false` if they are a conservative over-approximation. The latter means that
/// `effects` contain instances not associated with a specific value.
static bool
getEffectsBefore(Operation *op,
                 SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
                 bool stopAtBarrier) {
  if (!op->getBlock())
    return true;

  // If there is a non-structured control flow, bail.
  Region *region = op->getBlock()->getParent();
  if (region && !llvm::hasSingleElement(region->getBlocks())) {
    addAllValuelessEffects(effects);
    return false;
  }

  // Collect all effects before the op.
  if (op != &op->getBlock()->front()) {
    for (Operation *it = op->getPrevNode(); it != nullptr;
         it = it->getPrevNode()) {
      if (isa<mlir::gpu::BarrierOp>(it)) {
        if (stopAtBarrier)
          return true;
        else
          continue;
      }
      if (!collectEffects(it, effects))
        return false;
    }
  }

  // Stop if reached the parallel region boundary.
  if (isParallelRegionBoundary(op->getParentOp()))
    return true;

  // Otherwise, keep collecting above the parent operation.
  if (!getEffectsBefore(op->getParentOp(), effects, stopAtBarrier))
    return false;

  // If the op is loop-like, collect effects from the trailing operations until
  // we hit a barrier because they can executed before the current operation by
  // the previous iteration of this loop. For example, in the following loop
  //
  //   for i = ... {
  //     op1
  //     ...
  //     barrier
  //     op2
  //   }
  //
  // the operation `op2` at iteration `i` is known to be executed before the
  // operation `op1` at iteration `i+1` and the side effects must be ordered
  // appropriately.
  if (isSequentialLoopLike(op->getParentOp())) {
    // Assuming loop terminators have no side effects.
    return getEffectsBefore(op->getBlock()->getTerminator(), effects,
                            /*stopAtBarrier=*/true);
  }

  // If the parent operation is not guaranteed to execute its (single-block)
  // region once, walk the block.
  bool conservative = false;
  if (!hasSingleExecutionBody(op->getParentOp()))
    op->getParentOp()->walk([&](Operation *in) {
      if (conservative)
        return WalkResult::interrupt();
      if (!collectEffects(in, effects)) {
        conservative = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

  return !conservative;
}

/// Collects memory effects from operations that may be executed after `op` in
/// a trivial structured control flow, e.g., without branches. Stops at the
/// parallel region boundary or at the barrier operation if `stopAtBarrier` is
/// set. Returns `true` if the memory effects added to `effects` are exact,
/// `false` if they are a conservative over-approximation. The latter means that
/// `effects` contain instances not associated with a specific value.
static bool
getEffectsAfter(Operation *op,
                SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
                bool stopAtBarrier) {
  if (!op->getBlock())
    return true;

  // If there is a non-structured control flow, bail.
  Region *region = op->getBlock()->getParent();
  if (region && !llvm::hasSingleElement(region->getBlocks())) {
    addAllValuelessEffects(effects);
    return false;
  }

  // Collect all effects after the op.
  if (op != &op->getBlock()->back())
    for (Operation *it = op->getNextNode(); it != nullptr;
         it = it->getNextNode()) {
      if (isa<mlir::gpu::BarrierOp>(it)) {
        if (stopAtBarrier)
          return true;
        continue;
      }
      if (!collectEffects(it, effects))
        return false;
    }

  // Stop if reached the parallel region boundary.
  if (isParallelRegionBoundary(op->getParentOp()))
    return true;

  // Otherwise, keep collecting below the parent operation.
  if (!getEffectsAfter(op->getParentOp(), effects, stopAtBarrier))
    return false;

  // If the op is loop-like, collect effects from the leading operations until
  // we hit a barrier because they can executed after the current operation by
  // the next iteration of this loop. For example, in the following loop
  //
  //   for i = ... {
  //     op1
  //     ...
  //     barrier
  //     op2
  //   }
  //
  // the operation `op1` at iteration `i` is known to be executed after the
  // operation `op2` at iteration `i-1` and the side effects must be ordered
  // appropriately.
  if (isSequentialLoopLike(op->getParentOp())) {
    if (isa<mlir::gpu::BarrierOp>(op->getBlock()->front()))
      return true;

    bool exact = collectEffects(&op->getBlock()->front(), effects);
    return getEffectsAfter(&op->getBlock()->front(), effects,
                           /*stopAtBarrier=*/true) &&
           exact;
  }

  // If the parent operation is not guaranteed to execute its (single-block)
  // region once, walk the block.
  bool conservative = false;
  if (!hasSingleExecutionBody(op->getParentOp()))
    op->getParentOp()->walk([&](Operation *in) {
      if (conservative)
        return WalkResult::interrupt();
      if (!collectEffects(in, effects)) {
        conservative = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

  return !conservative;
}

/// Looks through known "view-like" ops to find the base memref.
static Value getBase(Value v) {
  while (true) {
    Operation *definingOp = v.getDefiningOp();
    if (!definingOp)
      break;

    bool shouldContinue =
        TypeSwitch<Operation *, bool>(v.getDefiningOp())
            .Case<memref::CastOp, memref::SubViewOp, memref::ViewOp>(
                [&](auto op) {
                  v = op.getSource();
                  return true;
                })
            .Case<memref::TransposeOp>([&](auto op) {
              v = op.getIn();
              return true;
            })
            .Case<memref::CollapseShapeOp, memref::ExpandShapeOp>([&](auto op) {
              v = op.getSrc();
              return true;
            })
            .Default([](Operation *) { return false; });
    if (!shouldContinue)
      break;
  }
  return v;
}

/// Returns `true` if the value is defined as a function argument.
static bool isFunctionArgument(Value v) {
  auto arg = dyn_cast<BlockArgument>(v);
  return arg && isa<FunctionOpInterface>(arg.getOwner()->getParentOp());
}

/// Returns the operand that the operation "propagates" through it for capture
/// purposes. That is, if the value produced by this operation is captured, then
/// so is the returned value.


static Value propagatesCapture(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
    .Case(
      [](ViewLikeOpInterface viewLike) { return viewLike.getViewSource(); })
    .Case([](CastOpInterface castLike) { return castLike->getOperand(0); })
    .Case([](memref::TransposeOp transpose) { return transpose.getIn(); })
    .Case<memref::ExpandShapeOp, memref::CollapseShapeOp>(
      [](auto op) { return op.getSrc(); })
    .Default([](Operation *) { return Value(); });
}

/// Returns `true` if the given operation is known to capture the given value,
/// `false` if it is known not to capture the given value, `nullopt` if neither
/// is known.
static std::optional<bool> getKnownCapturingStatus(Operation *op, Value v) {
  return llvm::TypeSwitch<Operation *, std::optional<bool>>(op)
      // Store-like operations don't capture the destination, but do capture
      // the value.
      .Case<memref::StoreOp, vector::TransferWriteOp>(
          [&](auto op) { return op.getValue() == v; })
      .Case<vector::StoreOp, vector::MaskedStoreOp>(
          [&](auto op) { return op.getValueToStore() == v; })
      // These operations are known not to capture.
      .Case([](memref::DeallocOp) { return false; })
      // By default, we don't know anything.
      .Default([](Operation *) { return std::nullopt; });
}

/// Returns `true` if the value may be captured by any of its users, i.e., if
/// the user may be storing this value into memory. This makes aliasing analysis
/// more conservative as it cannot assume the pointer-like value is only passed
/// around through SSA use-def.
static bool maybeCaptured(Value v) {
  SmallVector<Value> todo = {v};
  while (!todo.empty()) {
    Value v = todo.pop_back_val();
    for (Operation *user : v.getUsers()) {
      // A user that is known to only read cannot capture.
      auto iface = dyn_cast<MemoryEffectOpInterface>(user);
      if (iface) {
        SmallVector<MemoryEffects::EffectInstance> effects;
        iface.getEffects(effects);
        if (llvm::all_of(effects,
                         [](const MemoryEffects::EffectInstance &effect) {
                           return isa<MemoryEffects::Read>(effect.getEffect());
                         })) {
          continue;
        }
      }

      // When an operation is known to create an alias, consider if the
      // source is captured as well.
      if (Value v = propagatesCapture(user)) {
        todo.push_back(v);
        continue;
      }

      std::optional<bool> knownCaptureStatus = getKnownCapturingStatus(user, v);
      if (!knownCaptureStatus || *knownCaptureStatus)
        return true;
    }
  }

  return false;
}

/// Returns true if two values may be referencing aliasing memory. This is a
/// rather naive and conservative analysis. Values defined by different
/// allocation-like operations as well as values derived from those by casts and
/// views cannot alias each other. Similarly, values defined by allocations
/// inside a function cannot alias function arguments. Global values cannot
/// alias each other or local allocations. Values that are captured, i.e.
/// themselves potentially stored in memory, are considered as aliasing with
/// everything. This seems sufficient to achieve barrier removal in structured
/// control flow, more complex cases would require a proper dataflow analysis.
static bool mayAlias(Value first, Value second) {

  first = getBase(first);
  second = getBase(second);
  // Values derived from the same base memref do alias (unless we do a more
  // advanced analysis to prove non-overlapping accesses).
  if (first == second) {
    return true;
  }

  // Different globals cannot alias.
  if (auto globFirst = first.getDefiningOp<memref::GetGlobalOp>()) {
    if (auto globSecond = second.getDefiningOp<memref::GetGlobalOp>()) {
      return globFirst.getNameAttr() == globSecond.getNameAttr();
    }
  }
  // if (auto subSpanFirst =
  //         first.getDefiningOp<HAL::InterfaceBindingSubspanOp>()) {
  //   if (auto subSpanSecond =
  //           second.getDefiningOp<HAL::InterfaceBindingSubspanOp>()) {
  //     return subSpanFirst.getBindingAttr() == subSpanSecond.getBindingAttr();
  //   }
  // }

  bool isDistinct[] = {producesDistinctBase(first.getDefiningOp()),
                       producesDistinctBase(second.getDefiningOp())};
  bool isGlobal[] = {first.getDefiningOp<memref::GetGlobalOp>() != nullptr,
                     second.getDefiningOp<memref::GetGlobalOp>() != nullptr};

  // Non-equivalent distinct bases and globals cannot alias. At this point, we
  // have already filtered out based on values being equal and global name being
  // equal.
  if ((isDistinct[0] || isGlobal[0]) && (isDistinct[1] || isGlobal[1]))
    return false;

  bool isArg[] = {isFunctionArgument(first), isFunctionArgument(second)};

  // Distinct bases (allocations) cannot have been passed as an argument.
  if ((isDistinct[0] && isArg[1]) || (isDistinct[1] && isArg[0]))
    return false;

  // Non-captured base distinct values cannot conflict with another base value.
  if (isDistinct[0] && !maybeCaptured(first))
    return false;
  if (isDistinct[1] && !maybeCaptured(second))
    return false;

  // Otherwise, conservatively assume aliasing.
  return true;
}

/// Returns `true` if the effect may be affecting memory aliasing the value. If
/// the effect is not associated with any value, it is assumed to affect all
/// memory and therefore aliases with everything.
static bool mayAlias(MemoryEffects::EffectInstance a, Value v2) {
  if (Value v = a.getValue()) {
    return mayAlias(v, v2);
  }
  return true;
}

/// Returns `true` if the two effects may be affecting aliasing memory. If
/// an effect is not associated with any value, it is assumed to affect all
/// memory and therefore aliases with everything. Effects on different resources
/// cannot alias.
static bool mayAlias(MemoryEffects::EffectInstance a,
                     MemoryEffects::EffectInstance b) {
  if (a.getResource()->getResourceID() != b.getResource()->getResourceID())
    return false;
  if (Value v2 = b.getValue()) {
    return mayAlias(a, v2);
  } else if (Value v = a.getValue()) {
    return mayAlias(b, v);
  }
  return true;
}

/// Returns `true` if any of the "before" effect instances has a conflict with
/// any "after" instance for the purpose of barrier elimination. The effects are
/// supposed to be limited to a barrier synchronization scope. A conflict exists
/// if effects instances affect aliasing memory locations and at least on of
/// then as a write. As an exception, if the non-write effect is an allocation
/// effect, there is no conflict since we are only expected to see the
/// allocation happening in the same thread and it cannot be accessed from
/// another thread without capture (which we do handle in alias analysis).
static bool
haveConflictingEffects(ArrayRef<MemoryEffects::EffectInstance> beforeEffects,
                       ArrayRef<MemoryEffects::EffectInstance> afterEffects) {
  for (const MemoryEffects::EffectInstance &before : beforeEffects) {
    for (const MemoryEffects::EffectInstance &after : afterEffects) {
      // If cannot alias, definitely no conflict.
      if (!mayAlias(before, after))
        continue;

      // Read/read is not a conflict.
      if (isa<MemoryEffects::Read>(before.getEffect()) &&
          isa<MemoryEffects::Read>(after.getEffect())) {
        continue;
      }

      // Allocate/* is not a conflict since the allocation happens within the
      // thread context.
      // TODO: This is not the case for */Free unless the allocation happened in
      // the thread context, which we could also check for.
      if (isa<MemoryEffects::Allocate>(before.getEffect()) ||
          isa<MemoryEffects::Allocate>(after.getEffect())) {
        continue;
      }

      // In the particular case that the before effect is a free, we only have 2
      // possibilities:
      //   1. either the program is well-formed and there must be an interleaved
      //      alloc that must limit the scope of effect lookback and we can
      //      safely ignore the free -> read / free -> write and free -> free
      //      conflicts.
      //   2. either the program is ill-formed and we are in undefined behavior
      //      territory.
      if (isa<MemoryEffects::Free>(before.getEffect()))
        continue;

      // Other kinds of effects create a conflict, e.g. read-after-write.
      LLVM_DEBUG(
          DBGS() << "found a conflict between (before): " << before.getValue()
                 << " read:" << isa<MemoryEffects::Read>(before.getEffect())
                 << " write:" << isa<MemoryEffects::Write>(before.getEffect())
                 << " alloc:"
                 << isa<MemoryEffects::Allocate>(before.getEffect()) << " free:"
                 << isa<MemoryEffects::Free>(before.getEffect()) << "\n");
      LLVM_DEBUG(
          DBGS() << "and (after):                " << after.getValue()
                 << " read:" << isa<MemoryEffects::Read>(after.getEffect())
                 << " write:" << isa<MemoryEffects::Write>(after.getEffect())
                 << " alloc:" << isa<MemoryEffects::Allocate>(after.getEffect())
                 << " free:" << isa<MemoryEffects::Free>(after.getEffect())
                 << "\n");
      return true;
    }
  }

  return false;
}

namespace {
/// Barrier elimination pattern. If a barrier does not enforce any conflicting
/// pair of memory effects, including a pair that is enforced by another
/// barrier, it is unnecessary and can be removed. Adapted from
/// "High-Performance GPU-to-CPU Transpilation and Optimization via High-Level
/// Parallel Constructs" by Moses et.al. in PPoPP 2023 and implementation in
/// Polygeist.
class BarrierElimination final : public OpRewritePattern<mlir::gpu::BarrierOp> {
public:
  using OpRewritePattern<mlir::gpu::BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::gpu::BarrierOp barrier,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(DBGS() << "checking the necessity of: " << barrier << " "
                      << barrier.getLoc() << "\n");

    SmallVector<MemoryEffects::EffectInstance> beforeEffects;
    getEffectsBefore(barrier, beforeEffects, /*stopAtBarrier=*/true);

    SmallVector<MemoryEffects::EffectInstance> afterEffects;
    getEffectsAfter(barrier, afterEffects, /*stopAtBarrier=*/true);

    if (!haveConflictingEffects(beforeEffects, afterEffects)) {
      LLVM_DEBUG(DBGS() << "the surrounding barriers are sufficient, removing "
                        << barrier << "\n");
      rewriter.eraseOp(barrier);
      return success();
    }

    LLVM_DEBUG(DBGS() << "barrier is necessary: " << barrier << " "
                      << barrier.getLoc() << "\n");
    return failure();
  }
};
} // namespace

void buddy::gpu::EliminateGpuBarriersOp::build(OpBuilder &builder,
                                                      OperationState &state,
                                                      Value target) {
  build(builder, state, target.getType(), target);
}

DiagnosedSilenceableFailure
buddy::gpu::EliminateGpuBarriersOp::applyToOne(
    transform::TransformRewriter &rewriter, func::FuncOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  RewritePatternSet patterns(target.getContext());
  patterns.insert<BarrierElimination>(getContext());
  transform::ErrorCheckingTrackingListener listener(state, *this);
  auto checkErrors = llvm::make_scope_exit([&]() {
    // The TrackingListener API makes checking for errors mandatory. It is safe
    // to drop payload ops during this transform, so we can ignore all errors.
    (void)listener.checkAndResetError();
  });
  GreedyRewriteConfig config;
  config.listener = &listener;
  if (failed(
          applyPatternsAndFoldGreedily(target, std::move(patterns), config))) {
    return emitDefaultSilenceableFailure(target);
  }

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// SynchronizeLoopOp
//===----------------------------------------------------------------------===//

void buddy::gpu::SynchronizeLoopOp::getEffects(
  SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
transform::onlyReadsHandle(getForOp(), effects);
transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure buddy::gpu::SynchronizeLoopOp::applyToOne(
  transform::TransformRewriter &rewriter, scf::ForOp forOp,
  transform::ApplyToEachResultList &results,
  transform::TransformState &state) {
rewriter.setInsertionPointAfter(forOp);
rewriter.create<mlir::gpu::BarrierOp>(forOp.getLoc());
return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// CreateAsyncGroupsOp.
//===---------------------------------------------------------------------===//

void buddy::gpu::CreateAsyncGroupsOp::getEffects(
  SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
transform::onlyReadsHandle(getTarget(), effects);
transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure buddy::gpu::CreateAsyncGroupsOp::applyToOne(
  transform::TransformRewriter &rewriter, func::FuncOp target,
  transform::ApplyToEachResultList &results,
  transform::TransformState &state) {
createAsyncGroups(rewriter, cast<func::FuncOp>(target),
                                 getUseMmaSync());
return DiagnosedSilenceableFailure::success();
}

#define GET_OP_CLASSES
#include "GPU/TransformOps.cpp.inc"
