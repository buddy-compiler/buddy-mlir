//===- ConvertMemcpyToGPU.cpp ---------------------------------------------===//
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
// This file implements the pass that converts memcpy to gpu operations.
//
//===---------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

#include <vector>

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// ConvertMemcpyToGPUPass
//===----------------------------------------------------------------------===//

namespace {

class ConvertMemcpyToGPUPass
    : public PassWrapper<ConvertMemcpyToGPUPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertMemcpyToGPUPass)
  StringRef getArgument() const final { return "convert-memcpy-to-gpu"; }
  StringRef getDescription() const final {
    return "Convert memref opertaions to gpu operations.";
  }
  ConvertMemcpyToGPUPass() = default;
  ConvertMemcpyToGPUPass(const ConvertMemcpyToGPUPass &) {}

  Option<bool> processArgs{
      *this, "process-args",
      llvm::cl::desc("Whether the pass processes the input args."),
      llvm::cl::init(true)};

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, memref::MemRefDialect>();
  }
};

MemRefType stripMemRefLayout(const MemRefType &base) {
  return MemRefType::get(base.getShape(), base.getElementType(), AffineMap(),
                         base.getMemorySpace());
}

// Returns true if `v` is (directly or through view ops) an argument to a
// gpu::LaunchFuncOp.
static bool hasGPULaunchUser(Value v) {
  for (auto *user : v.getUsers()) {
    if (isa<gpu::LaunchFuncOp, func::CallOp>(user))
      return true;
    if (isa<memref::ExpandShapeOp, memref::CollapseShapeOp, memref::SubViewOp,
            memref::CastOp, memref::ReinterpretCastOp>(user)) {
      for (auto res : user->getResults())
        if (isa<MemRefType>(res.getType()) && hasGPULaunchUser(res))
          return true;
    }
  }
  return false;
}

// Returns true if `v` has any user that is NOT a GPU operation, memory
// deallocation, memcpy (which the pass will convert), or a pure view op.
// Such users are host-side (CPU) compute operations.
static bool hasCPUComputeUser(Value v) {
  for (auto *user : v.getUsers()) {
    if (isa<gpu::LaunchFuncOp, gpu::MemcpyOp, memref::DeallocOp, memref::CopyOp,
            func::CallOp, func::ReturnOp>(user))
      continue;
    if (isa<memref::ExpandShapeOp, memref::CollapseShapeOp, memref::SubViewOp,
            memref::CastOp, memref::ReinterpretCastOp>(user)) {
      for (auto res : user->getResults())
        if (isa<MemRefType>(res.getType()) && hasCPUComputeUser(res))
          return true;
      continue;
    }
    return true;
  }
  return false;
}

static bool isReturnedByFunction(Value v) {
  for (auto *user : v.getUsers()) {
    if (isa<func::ReturnOp>(user))
      return true;
    if (isa<memref::ExpandShapeOp, memref::CollapseShapeOp, memref::SubViewOp,
            memref::CastOp, memref::ReinterpretCastOp>(user)) {
      for (auto res : user->getResults())
        if (isa<MemRefType>(res.getType()) && isReturnedByFunction(res))
          return true;
    }
  }
  return false;
}

void ConvertMemcpyToGPUPass::runOnOperation() {
  auto funcOp = getOperation();

  if (funcOp.isDeclaration() || funcOp.isExternal())
    return;

  // Make sure the gpu function is already outlined.
  funcOp->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
    if (auto gpuLaunchOp = dyn_cast<gpu::LaunchOp>(nestedOp)) {
      nestedOp->emitOpError("The gpu function should be outlined.");
    }
    return WalkResult::advance();
  });

  std::vector<Value> unDeallocatedValue;
  // Maps original memref.alloc results to their gpu.alloc replacements.
  // Only populated for non-mixed (pure GPU) allocs where the original
  // memref.alloc was erased.
  DenseMap<Value, Value> erasedAllocToGpuMap;
  OpBuilder builder(funcOp->getContext());

  // Copy all function arguments to gpu, needs deallocation
  if (processArgs) {
    builder.setInsertionPointToStart(&(funcOp.getBody().front()));
    unsigned numArgs = funcOp.getNumArguments();
    for (unsigned i = 0; i < numArgs; ++i) {
      BlockArgument arg = funcOp.getArgument(i);
      // Create a gpu.alloc op, then copy memory to it
      // TODO: Move this out of operation, make the copy process async
      auto memrefType = dyn_cast<MemRefType>(arg.getType());
      if (!memrefType)
        continue;
      if (llvm::any_of(memrefType.getShape(), ShapedType::isDynamic) ||
          memrefType.getElementType().isIndex() ||
          memrefType.getElementType().isInteger(1))
        continue;
      // Skip strided (non-contiguous) args. gpu::MemcpyOp requires both
      // operands to be contiguous. Strided args typically point into managed
      // memory (e.g. param subviews) and are GPU-accessible as-is.
      if (!memrefType.getLayout().isIdentity())
        continue;

      auto contiguousType = stripMemRefLayout(memrefType);
      bool returned = isReturnedByFunction(arg);
      auto gpuAllocOp =
          gpu::AllocOp::create(builder, builder.getUnknownLoc(),
                               TypeRange({contiguousType}), ValueRange({}));
      if (!returned)
        unDeallocatedValue.push_back(gpuAllocOp->getResult(0));
      auto gpuMemcpyOp =
          gpu::MemcpyOp::create(builder, gpuAllocOp.getLoc(), TypeRange(),
                                ValueRange(), gpuAllocOp.getResult(0), arg);
      // If the arg has a non-identity layout (e.g. dynamic strides), the
      // gpu.alloc has a different type (identity layout). Cast back to the
      // original strided type so that gpu.launch_func and other users keep
      // the same operand type and pass verification.
      Value replacement = gpuAllocOp->getResult(0);
      if (contiguousType != memrefType)
        replacement = memref::CastOp::create(builder, gpuAllocOp.getLoc(),
                                             memrefType, replacement);
      arg.replaceAllUsesExcept(replacement, gpuMemcpyOp);
    }
  }

  funcOp->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
    // Replace all allocations with GPU.alloc
    if (auto allocOp = dyn_cast<memref::AllocOp>(nestedOp)) {
      // Rewrite this allocOp to gpu.alloc, change for all users
      builder.setInsertionPointAfter(allocOp);
      auto result = allocOp->getResult(0);
      auto memrefType = dyn_cast<MemRefType>(result.getType());
      auto memorySpace = memrefType.getMemorySpace();

      // Skip index-typed memrefs (bookkeeping buffers, not compute data).
      if (memrefType.getElementType().isIndex())
        return WalkResult::advance();

      // Filter operations.
      if (memorySpace) {
        if (auto intMemorySpace = llvm::dyn_cast<IntegerAttr>(memorySpace)) {
          if (intMemorySpace.getInt() != 0) {
            return WalkResult::advance();
          }
        } else if (auto gpuMemorySpace =
                       llvm::dyn_cast<gpu::AddressSpaceAttr>(memorySpace)) {
          if (gpuMemorySpace.getValue() != gpu::AddressSpace::Global) {
            return WalkResult::advance();
          }
        } else
          return WalkResult::advance();
      }

      // Note: we process ALL allocs to ensure they use managed memory,
      // even if not directly used by GPU kernels. Allocs that flow through
      // func::CallOp to GPU subgraphs need managed memory too.

      auto gpuAllocOp =
          gpu::AllocOp::create(builder, allocOp->getLoc(),
                               TypeRange({stripMemRefLayout(memrefType)}),
                               allocOp.getDynamicSizes());

      // Replace all uses with the GPU buffer. Since all gpu.alloc use
      // managed memory (cuMemAllocManaged), CPU code can also read/write
      // the GPU buffer directly — no need for a separate mixed path.
      erasedAllocToGpuMap[result] = gpuAllocOp.getResult(0);
      for (auto user : llvm::make_early_inc_range(result.getUsers())) {
        if (auto deallocOp = dyn_cast<memref::DeallocOp>(user)) {
          builder.setInsertionPointAfter(deallocOp);
          gpu::DeallocOp::create(builder, deallocOp->getLoc(), TypeRange(),
                                 ValueRange(), gpuAllocOp.getResult(0));
          deallocOp->erase();
        } else {
          for (auto &opOperand : user->getOpOperands()) {
            if (opOperand.is(result))
              opOperand.set(gpuAllocOp.getResult(0));
          }
        }
      }
      allocOp->erase();
    }
    // Replace memref.dealloc with gpu.dealloc for GPU-allocated buffers
    else if (auto deallocOp = dyn_cast<memref::DeallocOp>(nestedOp)) {
      auto memrefVal = deallocOp.getOperand();
      // Check if this memref was converted to a GPU allocation.
      // First check the mapping table (for non-mixed allocs whose original
      // memref.alloc was erased).
      Value gpuVal;
      auto it = erasedAllocToGpuMap.find(memrefVal);
      if (it != erasedAllocToGpuMap.end()) {
        gpuVal = it->second;
      } else if (auto gpuAllocOp = memrefVal.getDefiningOp<gpu::AllocOp>()) {
        gpuVal = memrefVal;
      } else if (auto castOp = memrefVal.getDefiningOp<memref::CastOp>()) {
        if (auto gpuAllocOp =
                castOp.getSource().getDefiningOp<gpu::AllocOp>()) {
          gpuVal = castOp.getSource();
        }
      }
      if (gpuVal) {
        builder.setInsertionPointAfter(deallocOp);
        gpu::DeallocOp::create(builder, deallocOp->getLoc(), TypeRange(),
                               ValueRange(), gpuVal);
        deallocOp->erase();
      }
    }
    // Replace all memory.copy operations with gpu.memcpy
    else if (auto copyOp = dyn_cast<memref::CopyOp>(nestedOp)) {
      auto src = copyOp.getOperand(0);
      auto dst = copyOp.getOperand(1);
      auto srcType = dyn_cast<MemRefType>(src.getType());
      auto dstType = dyn_cast<MemRefType>(dst.getType());
      // Skip non-contiguous (strided) copies. gpu::MemcpyOp requires both
      // operands to be contiguous. Strided copies are left as memref.copy and
      // will be lowered to @memrefCopy by the NVVM pipeline, running on the
      // CPU side (safe with unified memory).
      if ((srcType && !srcType.getLayout().isIdentity()) ||
          (dstType && !dstType.getLayout().isIdentity()))
        return WalkResult::advance();
      // Notice: GPU.memcpy has a different src dst order
      builder.setInsertionPointAfter(copyOp);
      gpu::MemcpyOp::create(builder, copyOp->getLoc(), TypeRange(),
                            ValueRange(), dst, src);
      copyOp->erase();
    }
    // Allocate/copy globals to GPU only when they are consumed by GPU kernels.
    // Keep pure CPU global uses untouched to avoid host load/store on GPU
    // pointers.
    else if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(nestedOp)) {
      auto result = getGlobalOp->getResult(0);
      auto memrefType = dyn_cast<MemRefType>(result.getType());
      builder.setInsertionPointAfter(getGlobalOp);
      bool returned = isReturnedByFunction(result);
      auto gpuAllocOp = gpu::AllocOp::create(
          builder, getGlobalOp->getLoc(),
          TypeRange({stripMemRefLayout(memrefType)}), ValueRange({}));
      if (!returned)
        unDeallocatedValue.push_back(gpuAllocOp->getResult(0));

      auto src = result;
      auto dst = gpuAllocOp->getResult(0);
      auto gpuMemcpyOp = gpu::MemcpyOp::create(
          builder, gpuAllocOp->getLoc(), TypeRange(), ValueRange(), dst, src);
      src.replaceAllUsesExcept(dst, gpuMemcpyOp);
    } else if (auto returnOp = dyn_cast<func::ReturnOp>(nestedOp)) {
      builder.setInsertionPoint(returnOp);
      llvm::SmallVector<Type> outputTypes(
          funcOp.getFunctionType().getResults());
      for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
        auto val = returnOp->getOperand(i);
        if (isa<MemRefType>(val.getType()))
          outputTypes[i] = val.getType();
      }
      for (auto value : unDeallocatedValue) {
        gpu::DeallocOp::create(builder, returnOp->getLoc(), TypeRange(),
                               ValueRange(), value);
      }
      funcOp.setType(
          builder.getFunctionType(funcOp.getArgumentTypes(), outputTypes));
    }
    return WalkResult::advance();
  });

  // CUDA limits gridY and gridZ to 65535. When gridY exceeds this limit and
  // gridX is within range, swap the two dimensions in both the launch and the
  // kernel body (gpu.block_id x ↔ y) so the large dimension uses gridX
  // (which supports up to 2^31-1).
  constexpr int64_t kMaxGridYZ = 65535;
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return;

  funcOp->walk([&](gpu::LaunchFuncOp launchOp) {
    Value gridX = launchOp.getGridSizeX();
    Value gridY = launchOp.getGridSizeY();
    auto cstY = dyn_cast_or_null<arith::ConstantIndexOp>(gridY.getDefiningOp());
    if (!cstY || cstY.value() <= kMaxGridYZ)
      return;
    auto cstX = dyn_cast_or_null<arith::ConstantIndexOp>(gridX.getDefiningOp());
    if (!cstX || cstX.value() > kMaxGridYZ)
      return;

    launchOp.getGridSizeXMutable().assign(gridY);
    launchOp.getGridSizeYMutable().assign(gridX);

    auto kernelModuleName = launchOp.getKernelModuleName();
    auto kernelName = launchOp.getKernelName();
    for (auto gpuModule : moduleOp.getOps<gpu::GPUModuleOp>()) {
      if (gpuModule.getName() != kernelModuleName)
        continue;
      if (auto kernelFunc =
              gpuModule.lookupSymbol<gpu::GPUFuncOp>(kernelName)) {
        kernelFunc.walk([](gpu::BlockIdOp blockIdOp) {
          if (blockIdOp.getDimension() == gpu::Dimension::x)
            blockIdOp.setDimension(gpu::Dimension::y);
          else if (blockIdOp.getDimension() == gpu::Dimension::y)
            blockIdOp.setDimension(gpu::Dimension::x);
        });
      }
    }
  });
}
} // end anonymous namespace.

namespace mlir {
namespace buddy {
void registerConvertMemcpyToGPUPass() {
  PassRegistration<ConvertMemcpyToGPUPass>();
}
} // namespace buddy
} // namespace mlir
