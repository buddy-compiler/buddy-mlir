//===- GPUHostRegister.cpp
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
// This file implements the GPU host register pass that adds gpu.host_register.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
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

class GPUHostRegisterPattern : public ConversionPattern {
public:
  explicit GPUHostRegisterPattern(MLIRContext *context)
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
// GPUHostRegisterPass
//===----------------------------------------------------------------------===//

namespace {

class GPUHostRegisterPass
    : public PassWrapper<GPUHostRegisterPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUHostRegisterPass)
  StringRef getArgument() const final { return "gpu-host-register"; }
  StringRef getDescription() const final {
    return "Register host memory to legalize gpu access.";
  }
  GPUHostRegisterPass() = default;
  GPUHostRegisterPass(const GPUHostRegisterPass &

  ) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, memref::MemRefDialect>();
  }
};
} // end anonymous namespace.

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
    }
    // else if (auto reallocOp)
    // else if (auto allocaOp)
    // Getglobal needs to create a copy
    else if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(producerOp)) {
      return {producerOp, 1};
    } 
    else if (auto subviewOp = dyn_cast<memref::SubViewOp>(producerOp)) {

    } 
    else if (auto loadOp = dyn_cast<memref::LoadOp>(producerOp)) {

    } 
    else if (auto collapseShapeOp =
                   dyn_cast<memref::CollapseShapeOp>(producerOp)) {

    } 
    else if (auto expandShapeOp =
                   dyn_cast<memref::ExpandShapeOp>(producerOp)) {

    } 
    else if (auto castOp = dyn_cast<memref::CastOp>(producerOp)) {

    } 
    else if (auto reinterpretCastOp =
                   dyn_cast<memref::ReinterpretCastOp>(producerOp)) {

    } 
    else if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(producerOp)) {

    } 
    else if (auto transposeOp = dyn_cast<memref::TransposeOp>(producerOp)) {

    } 
    else if (auto viewOp = dyn_cast<memref::ViewOp>(producerOp)) {

    } 
    else {
      llvm_unreachable("Unknown producer op");
    }
    // Look for parent op
    return {producerOp, 2};
  }
  llvm::dbgs() << "returning null:" << value << "\n";
  value->dump();
  // Values comes from outside the function
  return {nullptr, 3};
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

void GPUHostRegisterPass::runOnOperation() {
  auto module = getOperation();
  std::set<Operation *> allocations;
  std::map<Operation *, memref::AllocOp *> globalAllocations;
  module->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
    if (auto launchFuncOp = dyn_cast<gpu::LaunchFuncOp>(nestedOp)) {
      // OpBuilder barrierBuilder(launchFuncOp->getContext());
      // barrierBuilder.setInsertionPointAfter(launchFuncOp);
      // barrierBuilder.create<gpu::BarrierOp>(launchFuncOp->getLoc());

      for (auto operand : launchFuncOp->getOperands()) {
        if (!operand.getType().isa<BaseMemRefType>())
          continue;
        auto res = getAllocationOp(&operand);
        auto allocOp = res.first;
        Operation *insertionOp = nullptr;
        if (!allocOp)
          continue;
        if (res.second == 1) {
          // add a copy for this global op
          OpBuilder builder(allocOp->getContext());
          builder.setInsertionPointAfter(allocOp);
          auto memrefType = dyn_cast<MemRefType>(operand.getType());
          auto newAllocOp = builder.create<memref::AllocOp>(
              allocOp->getLoc(), memrefType, ValueRange{});
          builder.create<memref::CopyOp>(
              allocOp->getLoc(), allocOp->getResult(0), newAllocOp.getResult());
          for (size_t i = 0; i < launchFuncOp->getNumOperands(); i++) {
            if (launchFuncOp->getOperand(i) == operand) {
              launchFuncOp->setOperand(i, newAllocOp.getResult());
            }
          }
          auto result = allocations.insert(newAllocOp);
          auto elementType = memrefType.getElementType();
          UnrankedMemRefType resType = UnrankedMemRefType::get(elementType, 0);
          auto castOp = builder.create<memref::CastOp>(
              newAllocOp->getLoc(), resType, newAllocOp->getResult(0));
          builder.create<gpu::HostRegisterOp>(castOp->getLoc(),
                                              castOp.getResult());
          globalAllocations[allocOp] = &newAllocOp;
        }
        // else if (res.second == 2) {
        //   // From function outside
        //   // Insert at the beginning of the region
        //   OpBuilder builder(operand.getParentRegion());
        //   auto memrefType = dyn_cast<MemRefType>(operand.getType());
        //   auto newAllocOp = builder.create<memref::AllocOp>(
        //       allocOp->getLoc(), memrefType, ValueRange{});
        //   builder.create<memref::CopyOp>(
        //       allocOp->getLoc(), allocOp->getResult(0),
        //       newAllocOp.getResult());
        //   for (size_t i = 0; i < launchFuncOp->getNumOperands(); i++) {
        //     if (launchFuncOp->getOperand(i) == operand) {
        //       launchFuncOp->setOperand(i, newAllocOp.getResult());
        //     }
        //   }
        //   auto result = allocations.insert(newAllocOp);
        //   auto elementType = memrefType.getElementType();
        //   UnrankedMemRefType resType = UnrankedMemRefType::get(elementType,
        //   0); auto castOp = builder.create<memref::CastOp>(
        //       newAllocOp->getLoc(), resType, newAllocOp->getResult(0));
        //   builder.create<gpu::HostRegisterOp>(castOp->getLoc(),
        //                                       castOp.getResult());
        //   globalAllocations[allocOp] = &newAllocOp;
        // }
        else {
          insertionOp = allocOp;
          auto result = allocations.insert(insertionOp);
          if (result.second) {
            OpBuilder builder(insertionOp->getContext());
            builder.setInsertionPointAfter(insertionOp);
            auto memrefType = dyn_cast<MemRefType>(operand.getType());
            auto elementType = memrefType.getElementType();
            UnrankedMemRefType resType =
                UnrankedMemRefType::get(elementType, 0);
            Value cast = builder.create<memref::CastOp>(
                insertionOp->getLoc(), resType, insertionOp->getResult(0));
            builder.create<gpu::HostRegisterOp>(insertionOp->getLoc(), cast);
          } else {
            // llvm::dbgs() << insertionOp->getName().getStringRef()
            //              << " has been registered\n";
          }
        }
      }
      return WalkResult::advance();
    } else if (auto deallocOp = dyn_cast<memref::DeallocOp>(nestedOp)) {
      auto operand = deallocOp->getOperand(0);
      if (!operand.getType().isa<BaseMemRefType>())
        return WalkResult::advance();
      if (auto getGlobalOp =
              dyn_cast<memref::GetGlobalOp>(operand.getDefiningOp())) {
        if (globalAllocations.find(operand.getDefiningOp()) !=
            globalAllocations.end()) {
          auto allocOp = globalAllocations[operand.getDefiningOp()];
          deallocOp->setOperand(0, allocOp->getResult());
        }
      }
    }
    return WalkResult::advance();
  });
}

namespace mlir {
namespace buddy {
void registerGPUHostRegisterPass() { PassRegistration<GPUHostRegisterPass>(); }
} // namespace buddy
} // namespace mlir
