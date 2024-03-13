//===- ConvertMemcpyToGPU.cpp
//-------------------------------------------------===//
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
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
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

void ConvertMemcpyToGPUPass::runOnOperation() {
  auto funcOp = getOperation();

  // Make sure the gpu function is already outlined.
  funcOp->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
    if (auto gpuLaunchOp = dyn_cast<gpu::LaunchOp>(nestedOp)) {
      nestedOp->emitOpError("The gpu function should be outlined.");
    }
    return WalkResult::advance();
  });

  std::set<gpu::AllocOp *> unDeallocatedOperations;
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
      auto gpuAllocOp = builder.create<gpu::AllocOp>(
          builder.getUnknownLoc(), TypeRange({memrefType}), ValueRange({}));
      unDeallocatedOperations.insert(&gpuAllocOp);
      auto gpuMemcpyOp = builder.create<gpu::MemcpyOp>(
          gpuAllocOp.getLoc(), TypeRange(), ValueRange(),
          gpuAllocOp.getResult(0), arg);
      // Replace all users with GPU memory
      auto users = arg.getUsers();
      std::vector<Operation *> usersVec(users.begin(), users.end());
      for (auto user : usersVec) {
        // Don't replace memcpy's operand
        if (isa<gpu::MemcpyOp>(user))
          continue;
        for (size_t j = 0; j < user->getNumOperands(); j++) {
          if (user->getOperand(j) == arg) {
            user->setOperand(j, gpuAllocOp.getResult(0));
          }
        }
      }
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

      // Filter operations.
      if (memorySpace) {
        if (auto intMemorySpace = llvm::dyn_cast<IntegerAttr>(memorySpace)) {
          if (intMemorySpace.getInt() != 0) {
            return WalkResult::advance();
          }
        }
        else if (auto gpuMemorySpace = llvm::dyn_cast<gpu::AddressSpaceAttr>(memorySpace)){
          if (gpuMemorySpace.getValue()!=gpu::AddressSpace::Global) {
            return WalkResult::advance();
          }
        }
        else return WalkResult::advance();
      }

      auto gpuAllocOp = builder.create<gpu::AllocOp>(
          allocOp->getLoc(), TypeRange({memrefType}), ValueRange({}));
      auto users = result.getUsers();
      std::vector<Operation *> usersVec(users.begin(), users.end());
      for (auto user : usersVec) {
        for (size_t j = 0; j < user->getNumOperands(); j++) {
          // Only the return value will not have dealloc op
          if (auto deallocOp = dyn_cast<memref::DeallocOp>(user)) {
            builder.setInsertionPointAfter(deallocOp);
            auto gpuDeallocOp = builder.create<gpu::DeallocOp>(
                deallocOp->getLoc(), TypeRange(), ValueRange(),
                gpuAllocOp.getResult(0));
            deallocOp->erase();
          } else if (user->getOperand(j) == result) {
            user->setOperand(j, gpuAllocOp.getResult(0));
          }
        }
      }
      allocOp->erase();
    }
    // Replace all memory.copy operations with gpu.memcpy
    else if (auto copyOp = dyn_cast<memref::CopyOp>(nestedOp)) {
      auto src = copyOp.getOperand(0);
      auto dst = copyOp.getOperand(1);
      // Notice: GPU.memcpy has a different src dst order
      builder.setInsertionPointAfter(copyOp);
      auto gpuMemcpyOp = builder.create<gpu::MemcpyOp>(
          copyOp->getLoc(), TypeRange(), ValueRange(), dst, src);
      {
        auto users = src.getUsers();
        std::vector<Operation *> usersVec(users.begin(), users.end());
        for (auto user : usersVec) {
          for (size_t j = 0; j < user->getNumOperands(); j++) {
            if (user->getOperand(j) == src) {
              user->setOperand(j, gpuMemcpyOp.getOperand(1));
            }
          }
        }
      }
      {
        auto users = dst.getUsers();
        std::vector<Operation *> usersVec(users.begin(), users.end());
        for (auto user : usersVec) {
          for (size_t j = 0; j < user->getNumOperands(); j++) {
            if (user->getOperand(j) == src) {
              user->setOperand(j, gpuMemcpyOp.getOperand(0));
            }
          }
        }
      }
      copyOp->erase();
    }
    // Allocate space on GPU and copy global memrefs to GPU, needs deallocation
    else if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(nestedOp)) {
      builder.setInsertionPointAfter(getGlobalOp);
      auto result = getGlobalOp->getResult(0);
      auto memrefType = dyn_cast<MemRefType>(result.getType());
      auto gpuAllocOp = builder.create<gpu::AllocOp>(
          getGlobalOp->getLoc(), TypeRange({memrefType}), ValueRange({}));
      unDeallocatedOperations.insert(&gpuAllocOp);
      auto src = result;
      auto dst = gpuAllocOp->getResult(0);
      auto gpuMemcpyOp = builder.create<gpu::MemcpyOp>(
          gpuAllocOp->getLoc(), TypeRange(), ValueRange(), dst, src);
      {
        auto users = src.getUsers();
        std::vector<Operation *> usersVec(users.begin(), users.end());
        for (auto user : usersVec) {
          if (isa<gpu::MemcpyOp>(user))
            continue;
          // TODO: replace with src.replaceAllUsesExcept()
          for (size_t j = 0; j < user->getNumOperands(); j++) {
            if (user->getOperand(j) == src) {
              user->setOperand(j, dst);
            }
          }
        }
      }
    }
    // Copy data back to CPU, deallocate GPU, then return
    else if (auto returnOp = dyn_cast<func::ReturnOp>(nestedOp)) {
      builder.setInsertionPoint(returnOp);

      for (auto *gpuAllocOp : unDeallocatedOperations) {
        auto gpuDeallocOp = builder.create<gpu::DeallocOp>(
            builder.getUnknownLoc(), TypeRange(), ValueRange(),
            gpuAllocOp->getResult(0));
      }
      builder.setInsertionPoint(returnOp);
      for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
        auto val = returnOp->getOperand(i);
        auto memRefType = dyn_cast<MemRefType>(val.getType());
        auto allocOp = builder.create<memref::AllocOp>(builder.getUnknownLoc(),
                                                       memRefType);
        auto gpuMemcpyOp = builder.create<gpu::MemcpyOp>(
            allocOp.getLoc(), TypeRange(), ValueRange(), allocOp->getResult(0),
            val);
        auto gpuDeallocOp = builder.create<gpu::DeallocOp>(
            gpuMemcpyOp->getLoc(), TypeRange(), ValueRange(), val);
        returnOp->setOperand(i, allocOp->getResult(0));
      }
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
