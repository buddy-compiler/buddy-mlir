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
    if (isa<gpu::LaunchFuncOp>(user))
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
            func::CallOp>(user))
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
      // Skip bookkeeping memrefs (e.g. from @dealloc_helper): they use
      // dynamic shapes or index/i1 element types and must not go to GPU.
      if (llvm::any_of(memrefType.getShape(), ShapedType::isDynamic) ||
          memrefType.getElementType().isIndex() ||
          memrefType.getElementType().isInteger(1))
        continue;

      auto gpuAllocOp = builder.create<gpu::AllocOp>(
          builder.getUnknownLoc(), TypeRange({stripMemRefLayout(memrefType)}),
          ValueRange({}));
      unDeallocatedValue.push_back(gpuAllocOp->getResult(0));
      auto gpuMemcpyOp = builder.create<gpu::MemcpyOp>(
          gpuAllocOp.getLoc(), TypeRange(), ValueRange(),
          gpuAllocOp.getResult(0), arg);
      arg.replaceAllUsesExcept(gpuAllocOp->getResult(0), gpuMemcpyOp);
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

      // Skip allocs with no GPU launch_func users at all: these are purely
      // host-side buffers (e.g. reduction accumulators, scalar temporaries)
      // and must remain on CPU.
      if (!hasGPULaunchUser(result))
        return WalkResult::advance();

      auto gpuAllocOp = builder.create<gpu::AllocOp>(
          allocOp->getLoc(), TypeRange({stripMemRefLayout(memrefType)}),
          allocOp.getDynamicSizes());

      bool mixed = hasCPUComputeUser(result);

      if (!mixed) {
        // Pure GPU alloc: replace all uses with the GPU buffer.
        for (auto user : llvm::make_early_inc_range(result.getUsers())) {
          if (auto deallocOp = dyn_cast<memref::DeallocOp>(user)) {
            builder.setInsertionPointAfter(deallocOp);
            builder.create<gpu::DeallocOp>(deallocOp->getLoc(), TypeRange(),
                                           ValueRange(),
                                           gpuAllocOp.getResult(0));
            deallocOp->erase();
          } else {
            for (auto &opOperand : user->getOpOperands()) {
              if (opOperand.is(result))
                opOperand.set(gpuAllocOp.getResult(0));
            }
          }
        }
        allocOp->erase();
      } else {
        // Mixed CPU+GPU alloc: the buffer is used by both host code and GPU
        // kernels (e.g. a CPU reduction reads a buffer that a GPU kernel wrote,
        // or a GPU kernel reads a scalar that CPU code computed).
        // Strategy: keep the CPU alloc for host users; for each gpu.launch_func
        // that uses this alloc, insert cpu→gpu copy before the launch and
        // gpu→cpu copy after the launch so both sides see consistent data.
        unDeallocatedValue.push_back(gpuAllocOp.getResult(0));

        SmallVector<gpu::LaunchFuncOp> launchUsers;
        for (auto *user : result.getUsers())
          if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(user))
            launchUsers.push_back(launchOp);

        for (auto launchOp : launchUsers) {
          // Copy CPU→GPU before the kernel (handles CPU-write→GPU-read case).
          builder.setInsertionPoint(launchOp);
          builder.create<gpu::MemcpyOp>(launchOp->getLoc(), TypeRange(),
                                        ValueRange(), gpuAllocOp.getResult(0),
                                        result);
          // Redirect the kernel argument to the GPU buffer.
          for (auto &opOperand : launchOp->getOpOperands())
            if (opOperand.is(result))
              opOperand.set(gpuAllocOp.getResult(0));
          // Copy GPU→CPU after the kernel (handles GPU-write→CPU-read case).
          builder.setInsertionPointAfter(launchOp);
          builder.create<gpu::MemcpyOp>(launchOp->getLoc(), TypeRange(),
                                        ValueRange(), result,
                                        gpuAllocOp.getResult(0));
        }
        // CPU users (memref.load/store, scf.for bodies, etc.) keep using the
        // original CPU alloc — no changes needed for them.
      }
    }
    // Replace all memory.copy operations with gpu.memcpy
    else if (auto copyOp = dyn_cast<memref::CopyOp>(nestedOp)) {
      auto src = copyOp.getOperand(0);
      auto dst = copyOp.getOperand(1);
      // Notice: GPU.memcpy has a different src dst order
      builder.setInsertionPointAfter(copyOp);
      builder.create<gpu::MemcpyOp>(copyOp->getLoc(), TypeRange(), ValueRange(),
                                    dst, src);
      copyOp->erase();
    }
    // Allocate/copy globals to GPU only when they are consumed by GPU kernels.
    // Keep pure CPU global uses untouched to avoid host load/store on GPU
    // pointers.
    else if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(nestedOp)) {
      auto result = getGlobalOp->getResult(0);
      auto memrefType = dyn_cast<MemRefType>(result.getType());

      if (!hasGPULaunchUser(result))
        return WalkResult::advance();

      builder.setInsertionPointAfter(getGlobalOp);
      auto gpuAllocOp = builder.create<gpu::AllocOp>(
          getGlobalOp->getLoc(), TypeRange({stripMemRefLayout(memrefType)}),
          ValueRange({}));
      unDeallocatedValue.push_back(gpuAllocOp->getResult(0));

      bool mixed = hasCPUComputeUser(result);
      if (!mixed) {
        auto src = result;
        auto dst = gpuAllocOp->getResult(0);
        auto gpuMemcpyOp = builder.create<gpu::MemcpyOp>(
            gpuAllocOp->getLoc(), TypeRange(), ValueRange(), dst, src);
        src.replaceAllUsesExcept(dst, gpuMemcpyOp);
      } else {
        SmallVector<gpu::LaunchFuncOp> launchUsers;
        for (auto *user : result.getUsers())
          if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(user))
            launchUsers.push_back(launchOp);

        for (auto launchOp : launchUsers) {
          builder.setInsertionPoint(launchOp);
          builder.create<gpu::MemcpyOp>(launchOp->getLoc(), TypeRange(),
                                        ValueRange(), gpuAllocOp.getResult(0),
                                        result);
          for (auto &opOperand : launchOp->getOpOperands())
            if (opOperand.is(result))
              opOperand.set(gpuAllocOp.getResult(0));
          builder.setInsertionPointAfter(launchOp);
          builder.create<gpu::MemcpyOp>(launchOp->getLoc(), TypeRange(),
                                        ValueRange(), result,
                                        gpuAllocOp.getResult(0));
        }
      }
    }
    // Copy data back to CPU, deallocate GPU, then return
    else if (auto returnOp = dyn_cast<func::ReturnOp>(nestedOp)) {
      builder.setInsertionPoint(returnOp);
      llvm::SmallVector<Type> outputTypes(
          funcOp.getFunctionType().getResults());
      for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
        auto val = returnOp->getOperand(i);
        if (auto memrefType = dyn_cast<MemRefType>(val.getType())) {
          auto identityMemrefType = stripMemRefLayout(memrefType);
          auto allocOp = builder.create<memref::AllocOp>(returnOp->getLoc(),
                                                         identityMemrefType);
          builder.create<gpu::MemcpyOp>(allocOp.getLoc(), TypeRange(),
                                        ValueRange(), allocOp->getResult(0),
                                        val);
          // FIXME: may be leak memory
          // auto gpuDeallocOp = builder.create<gpu::DeallocOp>(
          //     gpuMemcpyOp->getLoc(), TypeRange(), ValueRange(), val);
          outputTypes[i] = identityMemrefType;
          returnOp->setOperand(i, allocOp->getResult(0));
        }
      }
      for (auto value : unDeallocatedValue) {
        builder.create<gpu::DeallocOp>(returnOp->getLoc(), TypeRange(),
                                       ValueRange(), value);
      }
      funcOp.setType(
          builder.getFunctionType(funcOp.getArgumentTypes(), outputTypes));
    }
    return WalkResult::advance();
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
