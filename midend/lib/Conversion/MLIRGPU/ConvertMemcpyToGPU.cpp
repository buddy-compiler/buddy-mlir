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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

#include <vector>

using namespace mlir;

//===----------------------------------------------------------------------===//
// ConvertMemcpyToGPUPass
//===----------------------------------------------------------------------===//

namespace {

class ConvertMemcpyToGPUPass
    : public PassWrapper<ConvertMemcpyToGPUPass,
                         InterfacePass<FunctionOpInterface>> {
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

void ConvertMemcpyToGPUPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();

  if (funcOp.isExternal())
    return;

  // Make sure the gpu function is already outlined.
  funcOp->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
    if (isa<gpu::LaunchOp>(nestedOp)) {
      nestedOp->emitOpError("The gpu function should be outlined.");
    }
    return WalkResult::advance();
  });

  std::vector<Value> unDeallocatedValue;
  IRRewriter rewriter(funcOp->getContext());

  // Copy all function arguments to gpu, needs deallocation
  if (processArgs) {
    rewriter.setInsertionPointToStart(&funcOp.front());
    unsigned numArgs = funcOp.getNumArguments();
    for (unsigned i = 0; i < numArgs; ++i) {
      BlockArgument arg = funcOp.getArgument(i);
      // Create a gpu.alloc op, then copy memory to it
      // TODO: Move this out of operation, make the copy process async
      auto memrefType = dyn_cast<MemRefType>(arg.getType());

      auto gpuAllocOp = gpu::AllocOp::create(
          rewriter, rewriter.getUnknownLoc(),
          TypeRange({stripMemRefLayout(memrefType)}), ValueRange({}));
      unDeallocatedValue.push_back(gpuAllocOp->getResult(0));
      auto gpuMemcpyOp =
          gpu::MemcpyOp::create(rewriter, gpuAllocOp.getLoc(), TypeRange(),
                                ValueRange(), gpuAllocOp.getResult(0), arg);
      arg.replaceAllUsesExcept(gpuAllocOp->getResult(0), gpuMemcpyOp);
    }
  }

  auto walkResult = funcOp->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
    // Replace all allocations with GPU.alloc
    if (auto allocOp = dyn_cast<memref::AllocOp>(nestedOp)) {
      // Rewrite this allocOp to gpu.alloc, change for all users
      rewriter.setInsertionPointAfter(allocOp);
      auto result = allocOp->getResult(0);
      auto memrefType = dyn_cast<MemRefType>(result.getType());
      auto memorySpace = memrefType.getMemorySpace();

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

      auto gpuAllocOp = gpu::AllocOp::create(
          rewriter, allocOp->getLoc(),
          TypeRange({stripMemRefLayout(memrefType)}), ValueRange({}));

      for (auto user : llvm::make_early_inc_range(result.getUsers())) {
        if (auto deallocOp = dyn_cast<memref::DeallocOp>(user)) {
          rewriter.setInsertionPointAfter(deallocOp);
          gpu::DeallocOp::create(rewriter, deallocOp->getLoc(), TypeRange(),
                                 ValueRange(), gpuAllocOp.getResult(0));
          rewriter.eraseOp(deallocOp);
        } else {
          for (auto &opOperand : user->getOpOperands()) {
            if (opOperand.is(result)) {
              opOperand.set(gpuAllocOp.getResult(0));
            }
          }
        }
      }
      rewriter.eraseOp(allocOp);
    }
    // Replace all memory.copy operations with gpu.memcpy
    else if (auto copyOp = dyn_cast<memref::CopyOp>(nestedOp)) {
      auto src = copyOp.getOperand(0);
      auto dst = copyOp.getOperand(1);
      auto srcType = dyn_cast<MemRefType>(src.getType());
      auto dstType = dyn_cast<MemRefType>(dst.getType());
      if (!srcType || !dstType) {
        copyOp.emitOpError("expected memref operands");
        return WalkResult::interrupt();
      }
      if (!srcType.getLayout().isIdentity() ||
          !dstType.getLayout().isIdentity()) {
        copyOp.emitOpError("strided memref.copy must be converted by "
                           "convert-strided-memref-copy-to-linalg before "
                           "convert-memcpy-to-gpu");
        return WalkResult::interrupt();
      }
      // Notice: GPU.memcpy has a different src dst order
      rewriter.setInsertionPointAfter(copyOp);
      gpu::MemcpyOp::create(rewriter, copyOp->getLoc(), TypeRange(),
                            ValueRange(), dst, src);
      rewriter.eraseOp(copyOp);
    }
    // Allocate space on GPU and copy global memrefs to GPU, needs deallocation
    else if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(nestedOp)) {
      rewriter.setInsertionPointAfter(getGlobalOp);
      auto result = getGlobalOp->getResult(0);
      auto memrefType = dyn_cast<MemRefType>(result.getType());
      auto gpuAllocOp = gpu::AllocOp::create(
          rewriter, getGlobalOp->getLoc(),
          TypeRange({stripMemRefLayout(memrefType)}), ValueRange({}));
      unDeallocatedValue.push_back(gpuAllocOp->getResult(0));

      auto src = result;
      auto dst = gpuAllocOp->getResult(0);
      auto gpuMemcpyOp = gpu::MemcpyOp::create(
          rewriter, gpuAllocOp->getLoc(), TypeRange(), ValueRange(), dst, src);
      src.replaceAllUsesExcept(dst, gpuMemcpyOp);
    }
    // Copy data back to CPU, deallocate GPU, then return
    else if (auto returnOp = dyn_cast<func::ReturnOp>(nestedOp)) {
      rewriter.setInsertionPoint(returnOp);
      auto fnType = cast<FunctionType>(funcOp.getFunctionType());
      llvm::SmallVector<Type> outputTypes(fnType.getResults());
      for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
        auto val = returnOp->getOperand(i);
        if (auto memrefType = dyn_cast<MemRefType>(val.getType())) {
          auto identityMemrefType = stripMemRefLayout(memrefType);
          auto allocOp = memref::AllocOp::create(rewriter, returnOp->getLoc(),
                                                 identityMemrefType);
          gpu::MemcpyOp::create(rewriter, allocOp.getLoc(), TypeRange(),
                                ValueRange(), allocOp->getResult(0), val);
          // FIXME: may be leak memory
          // auto gpuDeallocOp = rewriter.create<gpu::DeallocOp>(
          //     gpuMemcpyOp->getLoc(), TypeRange(), ValueRange(), val);
          outputTypes[i] = identityMemrefType;
          returnOp->setOperand(i, allocOp->getResult(0));
        }
      }
      for (auto value : unDeallocatedValue) {
        gpu::DeallocOp::create(rewriter, returnOp->getLoc(), TypeRange(),
                               ValueRange(), value);
      }
      funcOp.setType(
          rewriter.getFunctionType(funcOp.getArgumentTypes(), outputTypes));
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return signalPassFailure();
}
} // end anonymous namespace.

namespace mlir {
namespace buddy {
void registerConvertMemcpyToGPUPass() {
  PassRegistration<ConvertMemcpyToGPUPass>();
}
} // namespace buddy
} // namespace mlir
