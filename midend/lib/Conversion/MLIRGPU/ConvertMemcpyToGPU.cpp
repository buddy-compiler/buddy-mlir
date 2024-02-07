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
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class ConvertMemcpyToGPUPattern : public ConversionPattern {
public:
  explicit ConvertMemcpyToGPUPattern(MLIRContext *context)
      : ConversionPattern(gpu::LaunchFuncOp().getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << op->getName().getStringRef() << "\n";
    return success();
  }

private:
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ConvertMemcpyToGPUPass
//===----------------------------------------------------------------------===//

namespace {

class ConvertMemcpyToGPUPass
    : public PassWrapper<ConvertMemcpyToGPUPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertMemcpyToGPUPass)
  StringRef getArgument() const final { return "convert-memcpy-to-gpu"; }
  StringRef getDescription() const final {
    return "Convert memref opertaions to gpu operations.";
  }
  ConvertMemcpyToGPUPass() = default;
  ConvertMemcpyToGPUPass(const ConvertMemcpyToGPUPass &

  ) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, memref::MemRefDialect>();
  }
};

Value *getSourceOperand(Operation *op) {
  auto operands = op->getOperands();
  Value *memrefOperand = nullptr;
  for (auto operand : operands) {
    if (!operand.getType().isa<BaseMemRefType>())
      continue;
    if (memrefOperand) {
      llvm_unreachable("Op has more than one memref operand");
    }
    memrefOperand = &operand;
  }
  if (!memrefOperand) {
    llvm_unreachable("Op has no memref operand");
  }
  return memrefOperand;
}

std::pair<Operation *, int> getAllocationOp(Value *value) {
  if (auto *producerOp = value->getDefiningOp()) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(producerOp)) {
      // llvm::dbgs()<<allocOp->getName().getStringRef()<<":"<<allocOp<<"\n";
      // llvm::dbgs()<<"returning value:"<<allocOp->getResult(0)<<"\n";
      // llvm::dbgs()<<"returning location:"<<allocOp->getLoc()<<"\n";
      return {producerOp, 0};
    } else if (auto gpuAllocOp = dyn_cast<gpu::AllocOp>(producerOp)) {
      return {producerOp, 5};
    }
    // else if (auto reallocOp)
    // else if (auto allocaOp)
    // Getglobal needs to create a copy
    else if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(producerOp)) {
      return {producerOp, 1};
    } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(producerOp)) {
      for (auto operand : producerOp->getOperands()) {
        if (!operand.getType().isa<BaseMemRefType>())
          continue;
        return getAllocationOp(&operand);
      }
    } else if (auto collapseShapeOp =
                   dyn_cast<memref::CollapseShapeOp>(producerOp)) {
      for (auto operand : producerOp->getOperands()) {
        if (!operand.getType().isa<BaseMemRefType>())
          continue;
        return getAllocationOp(&operand);
      }
    } else if (auto expandShapeOp =
                   dyn_cast<memref::ExpandShapeOp>(producerOp)) {
      for (auto operand : producerOp->getOperands()) {
        if (!operand.getType().isa<BaseMemRefType>())
          continue;
        return getAllocationOp(&operand);
      }
    } else if (auto castOp = dyn_cast<memref::CastOp>(producerOp)) {
      for (auto operand : producerOp->getOperands()) {
        if (!operand.getType().isa<BaseMemRefType>())
          continue;
        return getAllocationOp(&operand);
      }
    } else if (auto reinterpretCastOp =
                   dyn_cast<memref::ReinterpretCastOp>(producerOp)) {
      for (auto operand : producerOp->getOperands()) {
        if (!operand.getType().isa<BaseMemRefType>())
          continue;
        return getAllocationOp(&operand);
      }
    } else if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(producerOp)) {
      for (auto operand : producerOp->getOperands()) {
        if (!operand.getType().isa<BaseMemRefType>())
          continue;
        return getAllocationOp(&operand);
      }
    } else if (auto transposeOp = dyn_cast<memref::TransposeOp>(producerOp)) {
      for (auto operand : producerOp->getOperands()) {
        if (!operand.getType().isa<BaseMemRefType>())
          continue;
        return getAllocationOp(&operand);
      }
    } else if (auto viewOp = dyn_cast<memref::ViewOp>(producerOp)) {
      for (auto operand : producerOp->getOperands()) {
        if (!operand.getType().isa<BaseMemRefType>())
          continue;
        return getAllocationOp(&operand);
      }
    } else if (auto arithConstantOp = dyn_cast<arith::ConstantOp>(producerOp)) {
      return {producerOp, 4};
    } else {
      llvm_unreachable("Unknown producer op");
    }
    // Look for parent op
  }
  // llvm::dbgs() << "returning null:" << value << "\n";
  // value->dump();
  // Values comes from outside the function
  return {reinterpret_cast<Operation *>(value), 3};
}
static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
  auto *lhs = const_cast<Operation *>(lhsC);
  auto *rhs = const_cast<Operation *>(rhsC);
  if (lhs == rhs)
    return true;

  return OperationEquivalence::isEquivalentTo(const_cast<Operation *>(lhsC),
                                              const_cast<Operation *>(rhsC),
                                              OperationEquivalence::None);
}

void ConvertMemcpyToGPUPass::runOnOperation() {
  auto module = getOperation();
  std::set<Operation *> allocations;
  std::map<Operation *, gpu::AllocOp *> gpuAllocations;
  std::map<Operation *, Operation *> globalAllocations;
  std::set<Value *> outsideValues;
  int launch_func = 0;
  module->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
    if (auto launchFuncOp = dyn_cast<gpu::LaunchFuncOp>(nestedOp)) {
      // OpBuilder barrierBuilder(launchFuncOp->getContext());
      // barrierBuilder.setInsertionPointAfter(launchFuncOp);
      // barrierBuilder.create<gpu::BarrierOp>(launchFuncOp->getLoc());

      for (size_t i = 0; i < launchFuncOp.getNumOperands(); i++) {
        auto operand = launchFuncOp.getOperand(i);
        if (!operand.getType().isa<BaseMemRefType>())
          continue;
        auto res = getAllocationOp(&operand);
        auto allocOp = res.first;
        if (!allocOp)
          continue;

        // Found Alloc OP
        if (res.second == 0) {
          auto result = allocations.insert(allocOp);
          // Newly created op for the allocation, add copy as well
          if (result.second) {
            // get memref types TODO: Use GPU memref type addrspace(1)
            OpBuilder builder(allocOp->getContext());
            builder.setInsertionPointAfter(allocOp);
            auto memrefType =
                dyn_cast<MemRefType>(allocOp->getResult(0).getType());
            auto elementType = memrefType.getElementType();
            // create a gpu.alloc op
            auto gpuAllocOp = builder.create<gpu::AllocOp>(
                allocOp->getLoc(), TypeRange({memrefType}), ValueRange({}));
            gpuAllocations[allocOp] = &gpuAllocOp;
            // replace the first-level users' (launch_func, expand_shape,
            // collapse_shape, etc.) operands
            auto users = allocOp->getUsers();
            // copy the users to a vector to avoid iterator invalidation
            std::vector<Operation *> usersVec(users.begin(), users.end());
            for (auto user : usersVec) {
              if (auto memcpyOp = dyn_cast<memref::CopyOp>(user)) {
                // find the memcpy op for cpu memory, then add a gpu memcpy op
                // after it
                builder.setInsertionPointAfter(memcpyOp);
                auto gpuMemcpyOp = builder.create<gpu::MemcpyOp>(
                    allocOp->getLoc(), TypeRange(), ValueRange(),
                    gpuAllocOp.getResult(0), allocOp->getResult(0));
              } else if (auto deallocOp = dyn_cast<memref::DeallocOp>(user)) {
                // adds a gpu dealloc op after the dealloc op for cpu memory
                builder.setInsertionPointAfter(deallocOp);
                auto gpuDeallocOp = builder.create<gpu::DeallocOp>(
                    allocOp->getLoc(), TypeRange(), ValueRange(),
                    gpuAllocOp.getResult(0));
              } else if (auto userLaunchFuncOp =
                             dyn_cast<gpu::LaunchFuncOp>(user)) {
                // real consumer for the allocation
                for (size_t j = 0; j < user->getNumOperands(); j++) {
                  if (user->getOperand(j) == allocOp->getResult(0)) {
                    user->setOperand(j, gpuAllocOp.getResult(0));
                  }
                }
              } else {
                // could be subview, collapse_shape, expand_shape, etc.
                // needs to find recursively for these ops and add gpu.memcpy
                // operations
                std::function<void(Operation *)> addMemcpy =
                    [&](Operation *op) -> void {
                  auto users = op->getUsers();
                  std::vector<Operation *> usersVec(users.begin(), users.end());
                  for (auto user : usersVec) {
                    if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
                      addMemcpy(user);
                    } else if (auto collapseShapeOp =
                                   dyn_cast<memref::CollapseShapeOp>(user)) {
                      addMemcpy(user);
                    } else if (auto expandShapeOp =
                                   dyn_cast<memref::ExpandShapeOp>(user)) {
                      addMemcpy(user);
                    } else if (auto castOp = dyn_cast<memref::CastOp>(user)) {
                      addMemcpy(user);
                    } else if (auto reinterpretCastOp =
                                   dyn_cast<memref::ReinterpretCastOp>(user)) {
                      addMemcpy(user);
                    } else if (auto reshapeOp =
                                   dyn_cast<memref::ReshapeOp>(user)) {
                      addMemcpy(user);
                    } else if (auto transposeOp =
                                   dyn_cast<memref::TransposeOp>(user)) {
                      addMemcpy(user);
                    } else if (auto memcpyOp = dyn_cast<memref::CopyOp>(user)) {
                      builder.setInsertionPointAfter(memcpyOp);
                      // Use gpu memcpy op and remove the cpu memcpy op
                      auto cpyDest = memcpyOp->getOperand(1);

                      auto gpuMemcpyOp = builder.create<gpu::MemcpyOp>(
                          allocOp->getLoc(), TypeRange(), ValueRange(), allocDest ,
                          memcpyOp->getOperand(0));
                      memcpyOp->erase();
                    }
                  }
                };
                addMemcpy(user);
              }
            }
          }
          // Already created op for the allocation
          else {
            // do nothing
          }
        }
        // Found global OP
        else if (res.second == 1) {

        }
        // Found value from outside of the function
        else if (res.second == 3) {
        }
        // Found arith.constant Op, needs host_register
        else if (res.second == 4) {

        }
        // Found GPU Alloc OP
        else if (res.second == 5) {
          continue;
        }
      }
      launch_func++;
      return WalkResult::advance();
    } else if (auto deallocOp = dyn_cast<memref::DeallocOp>(nestedOp)) {
      // do nothing
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
