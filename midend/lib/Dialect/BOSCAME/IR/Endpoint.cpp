//===- Endpoint.cpp - AME memory endpoint provenance -----------===//
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
// Implementation of endpoint provenance for BOSCAME planning and verification.
//
// Walks view-like ops back to a storage root, classifies ExternalDDR /
// PrivateAllocation / PrivateDDRFrame / SPM, and verifies the bounded private
// DDR frame (`bosc_ame.ddr.frame` / `bosc_ame.ddr.release`) plus alias escape
// rules.  Used by DMA and tile verifiers; never invents a root from element
// type alone.  See `Endpoint.h` for the public API.
//
//===----------------------------------------------------------------------===//

#include "Dialect/BOSCAME/Endpoint.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
using namespace mlir;
using namespace buddy::boscame;
Value buddy::boscame::getEndpointRoot(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (auto view = dyn_cast<ViewLikeOpInterface>(op))
      value = view.getViewSource();
    else if (auto metadata = dyn_cast<memref::ExtractStridedMetadataOp>(op))
      value = metadata.getSource();
    else
      break;
  }
  return value;
}
EndpointKind buddy::boscame::classifyEndpoint(Value value) {
  auto type = dyn_cast<BaseMemRefType>(value.getType());
  if (!type)
    return EndpointKind::Unknown;
  if (isa_and_nonnull<SPMAttr>(type.getMemorySpace()))
    return EndpointKind::SPM;
  if (type.getMemorySpace())
    return EndpointKind::Unknown;
  Value root = getEndpointRoot(value);
  if (auto argument = dyn_cast<BlockArgument>(root))
    if (argument.getOwner()->isEntryBlock() &&
        isa<func::FuncOp>(argument.getOwner()->getParentOp()))
      return EndpointKind::ExternalDDR;
  if (root.getDefiningOp<DDRFrameOp>())
    return EndpointKind::PrivateDDRFrame;
  if (root.getDefiningOp<memref::AllocOp>() ||
      root.getDefiningOp<memref::AllocaOp>())
    return EndpointKind::PrivateAllocation;
  return EndpointKind::Unknown;
}
static std::optional<int64_t>
storageBytes(MemRefType type, std::optional<int64_t> actualOffset = {}) {
  if (!type || !type.hasStaticShape())
    return std::nullopt;
  int64_t width = spmdma::getSpmElementByteWidth(type.getElementType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (!width || failed(type.getStridesAndOffset(strides, offset)))
    return std::nullopt;
  if (actualOffset)
    offset = *actualOffset;
  if (offset < 0)
    return std::nullopt;
  int64_t elements = offset;
  for (auto [size, stride] : llvm::zip(type.getShape(), strides)) {
    int64_t part;
    if (size <= 0 || stride <= 0 || llvm::MulOverflow(size - 1, stride, part) ||
        llvm::AddOverflow(elements, part, elements))
      return std::nullopt;
  }
  if (llvm::AddOverflow(elements, int64_t(1), elements) ||
      llvm::MulOverflow(elements, width, elements))
    return std::nullopt;
  return elements;
}
std::optional<int64_t> buddy::boscame::getStaticStorageBytes(Value value) {
  return storageBytes(dyn_cast<MemRefType>(value.getType()));
}
static bool serialAncestors(Operation *op, func::FuncOp function) {
  for (Operation *parent = op->getParentOp(); parent && parent != function;
       parent = parent->getParentOp())
    if (!isa<scf::ForOp, scf::IfOp, scf::ExecuteRegionOp>(parent))
      return false;
  return true;
}
static LogicalResult verifyAliasBounds(Operation *diagnostic, Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  auto origin = spmdma::getStaticPointerOrigin(value);
  auto offset = spmdma::getStaticDescriptorOffset(value);
  auto capacity = getStaticStorageBytes(getEndpointRoot(value));
  auto extent = storageBytes(type, offset);
  int64_t end;
  if (!type || type.getMemorySpace() || !origin || !capacity || !offset ||
      !extent || origin->root != getEndpointRoot(value) ||
      origin->byteShift < 0 ||
      llvm::AddOverflow(origin->byteShift, *extent, end) || end > *capacity)
    return diagnostic->emitOpError("private DDR endpoint has an unproved view "
                                   "bound or exceeds frame capacity");
  return success();
}
LogicalResult buddy::boscame::verifyPrivateAllocation(Operation *diagnostic,
                                                      Value value) {
  Value root = getEndpointRoot(value);
  auto kind = classifyEndpoint(root);
  if (kind != EndpointKind::PrivateAllocation &&
      kind != EndpointKind::PrivateDDRFrame)
    return diagnostic->emitOpError(
        "private DDR requires an actual allocation or managed frame root");
  Operation *allocation = root.getDefiningOp();
  auto function = allocation->getParentOfType<func::FuncOp>();
  auto bytes = getStaticStorageBytes(root);
  if (!function || function.getBody().getBlocks().size() != 1 || !bytes ||
      *bytes > kPrivateDDRFrameMaxBytes ||
      !serialAncestors(allocation, function))
    return diagnostic->emitOpError("private DDR requires bounded static "
                                   "storage in a serial single-entry function");
  SmallVector<Value> pending{root};
  llvm::DenseSet<Value> seen;
  while (!pending.empty()) {
    Value alias = pending.pop_back_val();
    if (!seen.insert(alias).second)
      continue;
    // Frame ownership covers CPU views as well as native AME endpoints. A
    // forged CPU-only view must not be able to overwrite another frame slice
    // or the allocator's guards before a later native tile access.
    if (failed(verifyAliasBounds(diagnostic, alias)))
      return failure();
    for (Operation *user : alias.getUsers()) {
      if (user->getParentOfType<func::FuncOp>() != function ||
          !serialAncestors(user, function))
        return diagnostic->emitOpError(
            "private DDR frame may not escape its serial invocation");
      if (auto metadata = dyn_cast<memref::ExtractStridedMetadataOp>(user)) {
        pending.push_back(metadata.getBaseBuffer());
        continue;
      }
      if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
        if (view.getViewSource() != alias || user->getNumResults() != 1)
          return diagnostic->emitOpError(
              "unsupported private DDR alias operation");
        pending.push_back(user->getResult(0));
        continue;
      }
      if (isa<memref::DeallocOp>(user)) {
        if (kind == EndpointKind::PrivateAllocation)
          continue;
        return diagnostic->emitOpError("managed private DDR storage requires "
                                       "ddr.release, not memref.dealloc");
      }
      if (isa<memref::LoadOp, memref::CopyOp, memref::DimOp, linalg::LinalgOp,
              vector::LoadOp, vector::StoreOp, vector::TransferReadOp,
              vector::TransferWriteOp, DmaStartOp, TileLoadAOp, TileLoadBOp,
              TileLoadAccOp, TileStoreAccOp, MatrixCopyOp, DDRReleaseOp>(user))
        continue;
      if (auto store = dyn_cast<memref::StoreOp>(user)) {
        if (store.getMemRef() == alias)
          continue;
      }
      return diagnostic->emitOpError("private DDR pointer escapes or has an "
                                     "unsupported memory user: ")
             << user->getName();
    }
  }
  return success();
}
LogicalResult buddy::boscame::verifyPrivateEndpoint(Operation *diagnostic,
                                                    Value value) {
  if (failed(verifyPrivateAllocation(diagnostic, value)))
    return failure();
  return verifyAliasBounds(diagnostic, value);
}

LogicalResult DDRFrameOp::verify() {
  auto type = dyn_cast<MemRefType>(getFrame().getType());
  auto function = (*this)->getParentOfType<func::FuncOp>();
  if (!type || type.getMemorySpace() || type.getRank() != 1 ||
      !type.getElementType().isInteger(8) || !type.getLayout().isIdentity() ||
      !type.hasStaticShape() || type.getDimSize(0) <= 0 ||
      type.getDimSize(0) > kPrivateDDRFrameMaxBytes ||
      type.getDimSize(0) % kPrivateDDRFrameAlignment)
    return emitOpError(
        "requires a positive aligned static i8 frame of at most 256 KiB");
  if (!function || function.getBody().getBlocks().size() != 1 ||
      getOperation()->getBlock() != &function.getBody().front())
    return emitOpError("must begin a serial single-block function invocation");
  for (Operation &previous : getOperation()->getBlock()->getOperations()) {
    if (&previous == getOperation())
      break;
    if (!isa<arith::ConstantOp>(previous))
      return emitOpError(
          "must precede executable operations in the invocation");
  }
  SmallVector<DDRReleaseOp> releases;
  for (Operation *user : getFrame().getUsers())
    if (auto release = dyn_cast<DDRReleaseOp>(user))
      releases.push_back(release);
  if (releases.size() != 1 ||
      releases.front()->getBlock() != getOperation()->getBlock() ||
      !isa_and_nonnull<func::ReturnOp>(releases.front()->getNextNode()))
    return emitOpError(
        "requires exactly one release immediately before the function return");
  if (failed(verifyPrivateAllocation(*this, getFrame())))
    return failure();
  return success();
}
LogicalResult DDRReleaseOp::verify() {
  auto frame = getFrame().getDefiningOp<DDRFrameOp>();
  if (!frame)
    return emitOpError("must release an actual private DDR frame root");
  return frame.verify();
}
