//===- SPMPlacement.cpp - Static buffers for SPM computations -------------===//
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

#include "Conversion/BOSCAMEHelpers/BOSCAMEHelpers.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace buddy::boscame::spmdma;

namespace buddy::boscame::lowering {
namespace {

bool isView(Operation *op) {
  return isa<memref::SubViewOp, memref::ReinterpretCastOp, memref::CastOp,
             memref::CollapseShapeOp, memref::ExpandShapeOp>(op);
}

// A placement only changes a local buffer and its views. Escaping buffers and
// scalar accesses keep their ordinary allocation; consumers must then either
// explicitly materialize them or report that the combination is unsupported.
bool collectViews(Value root, SmallVectorImpl<Value> &values) {
  values.push_back(root);
  for (size_t i = 0; i < values.size(); ++i) {
    for (Operation *user : values[i].getUsers()) {
      if (isView(user)) {
        Value view = user->getResult(0);
        if (!isa<MemRefType>(view.getType()))
          return false;
        if (!llvm::is_contained(values, view))
          values.push_back(view);
      } else if (!isa<linalg::FillOp, linalg::GenericOp, linalg::ReduceOp,
                      linalg::MatmulOp, memref::CopyOp, memref::DeallocOp,
                      memref::DimOp, vector::LoadOp, vector::StoreOp>(user)) {
        return false;
      }
    }
  }
  return true;
}

bool connectedToSpm(ArrayRef<Value> values) {
  for (Value value : values)
    for (Operation *user : value.getUsers())
      if (isa<linalg::GenericOp, memref::CopyOp>(user))
        for (Value operand : user->getOperands())
          if (getSpmBank(operand.getType()))
            return true;
  return false;
}

} // namespace

LogicalResult planSpmComputationBuffers(ModuleOp module) {
  SmallVector<Operation *> allocations;
  module.walk([&](Operation *op) {
    if (isa<memref::AllocOp, memref::AllocaOp>(op))
      allocations.push_back(op);
  });
  Workspace workspace;
  bool changed;
  do {
    changed = false;
    for (Operation *&allocation : allocations) {
      if (!allocation)
        continue;
      Value root = allocation->getResult(0);
      auto type = cast<MemRefType>(root.getType());
      auto bytes = getContiguousZeroOffsetByteSize(type);
      // Rank-zero reduction accumulators intentionally remain ordinary scalar
      // storage. SPM allocation requests from bufferization must be resolved
      // even if no other SPM operand is present yet.
      bool alreadySpm = getSpmBank(type).has_value();
      if (!bytes || !type.hasStaticShape() || type.getRank() < 1 ||
          type.getRank() > 2 || *bytes <= 0 || *bytes % 64)
        continue;
      SmallVector<Value> views;
      if (!collectViews(root, views)) {
        if (alreadySpm)
          return allocation->emitError(
              "unsupported escaping or scalar SPM temporary");
        continue;
      }
      if (!alreadySpm && !connectedToSpm(views))
        continue;
      OpBuilder builder(allocation);
      auto storage = workspace.allocate(
          builder, allocation,
          MemRefType::get(type.getShape(), type.getElementType()),
          alreadySpm ? *getSpmBank(type) : -1);
      if (failed(storage))
        return failure();
      Attribute space = cast<MemRefType>(storage->getType()).getMemorySpace();
      for (Value view : llvm::drop_begin(views)) {
        auto old = cast<MemRefType>(view.getType());
        view.setType(MemRefType::get(old.getShape(), old.getElementType(),
                                     old.getLayout(), space));
      }
      SmallVector<memref::DeallocOp> deallocs;
      for (Value view : views)
        for (Operation *user : view.getUsers())
          if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
            deallocs.push_back(dealloc);
      for (memref::DeallocOp dealloc : deallocs)
        dealloc.erase();
      root.replaceAllUsesWith(*storage);
      allocation->erase();
      allocation = nullptr;
      changed = true;
    }
  } while (changed);
  return success();
}

} // namespace buddy::boscame::lowering
