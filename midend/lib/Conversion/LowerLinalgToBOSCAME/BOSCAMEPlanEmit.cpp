//===- BOSCAMEPlanEmit.cpp - Mechanically expand verified planned IR ------===//
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
// Mechanically expand a verified BOSCAME plan into concrete dialect ops,
// preserving task and source provenance without re-selecting routes.
//
//===----------------------------------------------------------------------===//

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEDDRPlan.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEFloatSchedule.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEMathApprox.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEOrderedMatmul.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEPlanIR.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMERegisterSchedule.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEStreamPlan.h"
#include "Dialect/BOSCAME/BOSCAMEAttrs.h"
#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "Dialect/BOSCAME/Endpoint.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "Dialect/BOSCAME/TileOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace ::buddy::boscame;

namespace {
DictionaryAttr plan(Operation *op, StringRef key) {
  return op->getAttrOfType<DictionaryAttr>(key);
}
DictionaryAttr nested(DictionaryAttr p, StringRef key) {
  return p.getAs<DictionaryAttr>(key);
}
int64_t number(DictionaryAttr p, StringRef key) {
  return p.getAs<IntegerAttr>(key).getInt();
}
StringRef text(DictionaryAttr p, StringRef key) {
  return p.getAs<StringAttr>(key).getValue();
}
int64_t elementBytes(Type type) { return type.getIntOrFloatBitWidth() / 8; }
Value index(OpBuilder &b, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(b, loc, value);
}

// This is descriptor reconstruction, not address analysis or route selection.
// reinterpret_cast replaces the descriptor offset; subview adds its shift.
// Keep dynamic SSA arithmetic visible and fold static descriptor arithmetic
// so the final dialect verifier can prove the selected SPM range.
Value descriptorOffset(OpBuilder &b, Location loc, Value source) {
  if (auto constant = spmdma::getStaticDescriptorOffset(source))
    return index(b, loc, *constant);
  if (auto view = source.getDefiningOp<memref::ReinterpretCastOp>()) {
    auto offset = view.getMixedOffsets().front();
    if (auto value = dyn_cast<Value>(offset))
      return value;
    return index(b, loc, cast<IntegerAttr>(cast<Attribute>(offset)).getInt());
  }
  if (auto view = source.getDefiningOp<memref::SubViewOp>()) {
    Value offset = descriptorOffset(b, loc, view.getSource());
    SmallVector<int64_t> strides;
    int64_t ignored;
    (void)cast<MemRefType>(view.getSource().getType())
        .getStridesAndOffset(strides, ignored);
    for (auto [part, stride] : llvm::zip(view.getMixedOffsets(), strides)) {
      Value term = dyn_cast<Value>(part);
      if (!term)
        term = index(b, loc, cast<IntegerAttr>(cast<Attribute>(part)).getInt());
      term = b.createOrFold<arith::MulIOp>(loc, term, index(b, loc, stride));
      offset = b.createOrFold<arith::AddIOp>(loc, offset, term);
    }
    return offset;
  }
  if (auto view = source.getDefiningOp<memref::CastOp>())
    return descriptorOffset(b, loc, view.getSource());
  if (auto view = source.getDefiningOp<memref::TransposeOp>())
    return descriptorOffset(b, loc, view.getIn());
  if (auto view = source.getDefiningOp<memref::CollapseShapeOp>())
    return descriptorOffset(b, loc, view.getSrc());
  if (auto view = source.getDefiningOp<memref::ExpandShapeOp>())
    return descriptorOffset(b, loc, view.getSrc());
  return memref::ExtractStridedMetadataOp::create(b, loc, source).getOffset();
}
Value endpoint(OpBuilder &b, Location loc, Value source, int64_t offsetBytes,
               int64_t countBytes) {
  auto sourceType = cast<MemRefType>(source.getType());
  int64_t width = elementBytes(sourceType.getElementType());
  Value offset = descriptorOffset(b, loc, source);
  if (offsetBytes)
    offset = b.createOrFold<arith::AddIOp>(loc, offset,
                                           index(b, loc, offsetBytes / width));
  auto type = MemRefType::get(
      {countBytes / width}, sourceType.getElementType(),
      StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic, {1}),
      sourceType.getMemorySpace());
  return memref::ReinterpretCastOp::create(
      b, loc, type, source, offset,
      ArrayRef<OpFoldResult>{b.getIndexAttr(countBytes / width)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
}
void moveFacts(Operation *source, Operation *completion) {
  for (NamedAttribute attr : source->getAttrs())
    if (attr.getName().getValue().starts_with("contract."))
      completion->setAttr(attr.getName(), attr.getValue());
}
Value tileSubview(OpBuilder &b, Location loc, Value source, int64_t row,
                  int64_t column, int64_t rows, int64_t columns) {
  return memref::SubViewOp::create(
      b, loc, source,
      ArrayRef<OpFoldResult>{b.getIndexAttr(row), b.getIndexAttr(column)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(rows), b.getIndexAttr(columns)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
}
void direct(Operation *op, DictionaryAttr selection, OpBuilder &b) {
  if (text(selection, "level") == "ddr")
    op->setAttr("bosc_ame.direct_ddr", b.getUnitAttr());
}

void materializeBuffers(ModuleOp module) {
  SmallVector<Operation *> allocations;
  module.walk([&](Operation *op) {
    if (isa<memref::AllocOp, memref::AllocaOp>(op))
      allocations.push_back(op);
  });
  DenseMap<Operation *, Value> frames;
  DenseMap<Operation *, int64_t> offsets;
  auto version = module->getAttrOfType<IntegerAttr>("plan.version");
  if (version && version.getInt() == 2) {
    // CPU boundary recipes still access external DDR when every AME private
    // buffer is in SPM. Their cache visibility must not depend on whether a
    // managed DDR frame happens to be needed by this physical placement.
    for (auto function : module.getOps<func::FuncOp>()) {
      bool hasAME = false, hasCpuDDR = false;
      function.walk([&](Operation *op) {
        if (auto route = plan(op, "plan.route"))
          hasAME |= text(route, "engine") == "ame";
        if (auto cpu = plan(op, "plan.cpu"))
          hasCpuDDR |= text(cpu, "engine") == "scalar-ddr";
      });
      if (hasAME && hasCpuDDR)
        module->setAttr("bosc_ame.cpu_ddr_policy",
                        UnitAttr::get(module.getContext()));
    }
    DenseSet<Value> frameRoots;
    module.walk([&](linalg::MatmulOp matmul) {
      if (auto route = plan(matmul, "plan.route")) {
        const StringRef names[] = {"a", "b", "accumulator"};
        for (auto [name, operand] : llvm::zip(names, matmul->getOperands())) {
          auto selected = nested(route, name);
          if (selected && text(selected, "level") == "ddr") {
            Value root = getEndpointRoot(operand);
            if (classifyEndpoint(root) == EndpointKind::PrivateAllocation)
              frameRoots.insert(root);
          }
        }
      }
    });
    module.walk([&](memref::CopyOp copy) {
      if (auto route = plan(copy, "plan.transfer")) {
        Value ddr;
        if (text(route, "from") == "ddr")
          ddr = copy.getSource();
        else if (text(route, "to") == "ddr")
          ddr = copy.getTarget();
        if (ddr && classifyEndpoint(ddr) == EndpointKind::PrivateAllocation)
          frameRoots.insert(getEndpointRoot(ddr));
      }
    });
    DenseMap<Operation *, int64_t> sizes;
    for (Operation *op : allocations) {
      auto p = plan(op, "plan.buffer");
      if (!p || text(p, "level") != "ddr" ||
          !frameRoots.contains(op->getResult(0)))
        continue;
      auto function = op->getParentOfType<func::FuncOp>();
      int64_t &size = sizes[function];
      offsets[op] = size;
      size +=
          llvm::alignTo(*getStaticStorageBytes(op->getResult(0)), int64_t(64));
    }
    for (auto &[owner, size] : sizes) {
      auto function = cast<func::FuncOp>(owner);
      OpBuilder b(&function.getBody().front(),
                  function.getBody().front().begin());
      auto frameType = MemRefType::get({size}, b.getI8Type());
      Value frame = DDRFrameOp::create(b, function.getLoc(), frameType);
      module->setAttr("bosc_ame.cpu_ddr_policy", b.getUnitAttr());
      frames[owner] = frame;
      b.setInsertionPoint(function.getBody().front().getTerminator());
      DDRReleaseOp::create(b, function.getLoc(), frame);
    }
  }
  for (Operation *op : allocations) {
    OpBuilder b(op);
    auto p = plan(op, "plan.buffer");
    if (!p)
      continue;
    auto type = cast<MemRefType>(op->getResult(0).getType());
    Value replacement;
    if (text(p, "level") == "spm") {
      int64_t bank = number(p, "bank");
      auto spmType =
          MemRefType::get(type.getShape(), type.getElementType(),
                          type.getLayout(), SPMAttr::get(b.getContext(), bank));
      replacement =
          SpmViewOp::create(b, op->getLoc(), spmType, b.getI32IntegerAttr(bank),
                            b.getI64IntegerAttr(number(p, "offset")));
    } else if (auto found = offsets.find(op); found != offsets.end()) {
      Value frame = frames[op->getParentOfType<func::FuncOp>()];
      auto storageType = type;
      int64_t storageOffset;
      SmallVector<int64_t> storageStrides;
      (void)type.getStridesAndOffset(storageStrides, storageOffset);
      if (!type.getLayout().isIdentity())
        storageType =
            MemRefType::get({*getStaticStorageBytes(op->getResult(0)) /
                             elementBytes(type.getElementType())},
                            type.getElementType());
      replacement = memref::ViewOp::create(
          b, op->getLoc(), storageType, frame,
          index(b, op->getLoc(), found->second), ValueRange{});
      if (storageType != type)
        replacement = memref::ReinterpretCastOp::create(
            b, op->getLoc(), type, replacement, storageOffset, type.getShape(),
            storageStrides);
    } else if (isa<memref::AllocOp>(op)) {
      replacement = memref::AllocaOp::create(b, op->getLoc(), type);
      if (Attribute alignment = op->getAttr("alignment"))
        replacement.getDefiningOp()->setAttr("alignment", alignment);
    } else {
      continue;
    }
    for (NamedAttribute attr : op->getAttrs())
      if (attr.getName().getValue().starts_with("contract."))
        replacement.getDefiningOp()->setAttr(attr.getName(), attr.getValue());
    SmallVector<Operation *> deallocations;
    SmallVector<Value> aliases{op->getResult(0)};
    DenseSet<Value> seen;
    while (!aliases.empty()) {
      Value alias = aliases.pop_back_val();
      if (!seen.insert(alias).second)
        continue;
      for (Operation *user : alias.getUsers()) {
        if (isa<memref::DeallocOp>(user))
          deallocations.push_back(user);
        else if (auto view = dyn_cast<ViewLikeOpInterface>(user);
                 view && view.getViewSource() == alias &&
                 user->getNumResults() == 1)
          aliases.push_back(user->getResult(0));
      }
    }
    for (Operation *dealloc : deallocations)
      dealloc->erase();
    op->getResult(0).replaceAllUsesWith(replacement);
    op->erase();
  }
  if (version && version.getInt() == 2) {
    module.walk([&](func::FuncOp function) {
      if (function.isExternal())
        return;
      SmallVector<memref::AllocaOp> scratch;
      function.walk([&](memref::AllocaOp op) {
        if (op->getNumOperands() == 0 && op.getType().hasStaticShape())
          scratch.push_back(op);
      });
      for (auto op : llvm::reverse(scratch)) {
        auto insertion = function.getBody().front().begin();
        if (auto found = frames.find(function); found != frames.end())
          insertion = std::next(found->second.getDefiningOp()->getIterator());
        op->moveBefore(&function.getBody().front(), insertion);
      }
    });
  }
  // All views preserve their shape and exact layout, changing only the
  // address space inherited from the already selected root placement.
  module.walk([&](Operation *op) {
    if (!isa<memref::SubViewOp, memref::ReinterpretCastOp, memref::CastOp,
             memref::ViewOp, memref::MemorySpaceCastOp, memref::TransposeOp,
             memref::CollapseShapeOp, memref::ExpandShapeOp>(op))
      return;
    auto source = dyn_cast<MemRefType>(op->getOperand(0).getType());
    auto result = dyn_cast<MemRefType>(op->getResult(0).getType());
    if (source && result)
      op->getResult(0).setType(
          MemRefType::get(result.getShape(), result.getElementType(),
                          result.getLayout(), source.getMemorySpace()));
  });
}

Operation *emitTransfer(memref::CopyOp copy) {
  OpBuilder b(copy);
  Location loc = copy.getLoc();
  auto p = plan(copy, "plan.transfer"), chunks = nested(p, "chunks");
  Operation *completion = nullptr;
  if (text(p, "engine") == "matrix") {
    // Configuration, offsets, exact extents and predecessor order have all
    // been verified. Each atomic operation owns its scratch accumulator and
    // completes before the next tile; emission makes no route decisions.
    for (Attribute attribute : p.getAs<ArrayAttr>("tiles")) {
      auto tile = cast<DictionaryAttr>(attribute);
      OperationState state(loc, MatrixCopyOp::getOperationName());
      state.addOperands({copy.getSource(), copy.getTarget()});
      for (StringRef name :
           {"source_offset", "target_offset", "rows", "row_bytes",
            "source_pitch", "target_pitch", "instruction_bits",
            "register_class", "register_count", "mtype"})
        state.addAttribute(name, tile.get(name));
      completion = b.create(state);
    }
    return completion;
  }
  // Boundary-only fields do not apply to a verified internal SPM copy. Both
  // matrix directions above use the recorded endpoint types and atomic tiles.
  bool load = text(p, "from") == "ddr";
  int64_t sourceStride = number(chunks, load ? "ddr_stride" : "spm_stride");
  int64_t targetStride = number(chunks, load ? "spm_stride" : "ddr_stride");
  int64_t size = number(chunks, "bytes"), count = number(chunks, "count");
  int64_t channel = number(p, "channel");
  if (text(p, "engine") == "segmented") {
    // The verifier has proved exact coverage, address legality, and this
    // predecessor order. Complete every segment before starting its successor;
    // no DMA request or RVV value survives into the next segment.
    for (Attribute attribute : p.getAs<ArrayAttr>("segments")) {
      auto segment = cast<DictionaryAttr>(attribute);
      int64_t segmentBytes = number(segment, "bytes");
      Value source = endpoint(b, loc, copy.getSource(),
                              number(segment, "source_offset"), segmentBytes);
      Value target = endpoint(b, loc, copy.getTarget(),
                              number(segment, "target_offset"), segmentBytes);
      if (text(segment, "engine") == "dma") {
        auto token = DmaStartOp::create(
            b, loc, DMATokenType::get(b.getContext(), channel), source, target,
            index(b, loc, segmentBytes), b.getI32IntegerAttr(channel));
        completion = DmaWaitOp::create(b, loc, token.getToken());
        continue;
      }
      auto type = cast<MemRefType>(source.getType());
      int64_t width = elementBytes(type.getElementType());
      int64_t lanes = number(segment, "vector_bytes") / width;
      auto loop = scf::ForOp::create(b, loc, index(b, loc, 0),
                                     index(b, loc, segmentBytes / width),
                                     index(b, loc, lanes));
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(loop.getBody());
        // Each vector's declared lane count is exact and divides this segment.
        // Fixed-vector legalization selects an adequate LMUL for the target's
        // minimum VLEN and sets AVL to these lanes, including short tails.
        auto loaded = vector::LoadOp::create(
            b, loc, VectorType::get({lanes}, type.getElementType()), source,
            ValueRange{loop.getInductionVar()});
        auto stored = vector::StoreOp::create(
            b, loc, loaded, target, ValueRange{loop.getInductionVar()});
        (load ? loaded.getOperation() : stored.getOperation())
            ->setAttr("bosc_ame.direct_ddr", b.getUnitAttr());
      }
      completion = loop;
    }
    return completion;
  }
  for (int64_t i = 0; i < count; ++i) {
    Value source = endpoint(b, loc, copy.getSource(), i * sourceStride, size);
    Value target = endpoint(b, loc, copy.getTarget(), i * targetStride, size);
    auto token = DmaStartOp::create(
        b, loc, DMATokenType::get(b.getContext(), channel), source, target,
        index(b, loc, size), b.getI32IntegerAttr(channel));
    completion = DmaWaitOp::create(b, loc, token.getToken());
  }
  return completion;
}

Operation *emitMatmul(linalg::MatmulOp matmul) {
  OpBuilder b(matmul);
  Location loc = matmul.getLoc();
  auto p = plan(matmul, "plan.route"), tile = nested(p, "tile"),
       sub = nested(p, "subtiles");
  int64_t m = number(tile, "m"), n = number(tile, "n"), k = number(tile, "k");
  auto emitTiles = [&](bool zero) -> Operation * {
    Operation *completion = nullptr;
    for (int64_t ni = 0; ni < number(sub, "n"); ++ni) {
      Value c = tileSubview(b, loc, matmul.getOutputs()[0], 0, ni * n, m, n);
      auto accType = VectorType::get({m, n}, b.getI32Type());
      Value accumulator;
      if (zero) {
        accumulator = TileZeroAccOp::create(b, loc, accType).getMatrix();
      } else {
        auto acc = TileLoadAccOp::create(b, loc, accType, c);
        direct(acc, nested(p, "accumulator"), b);
        accumulator = acc.getMatrix();
      }
      for (int64_t ki = 0; ki < number(sub, "k"); ++ki) {
        Value a = tileSubview(b, loc, matmul.getInputs()[0], 0, ki * k, m, k);
        Value operandB =
            tileSubview(b, loc, matmul.getInputs()[1], ki * k, ni * n, k, n);
        auto lhs = TileLoadAOp::create(
            b, loc, VectorType::get({m, k}, b.getI8Type()), a);
        auto rhs = TileLoadBOp::create(
            b, loc, VectorType::get({k, n}, b.getI8Type()), operandB);
        direct(lhs, nested(p, "a"), b);
        direct(rhs, nested(p, "b"), b);
        auto mma = TileMmaOp::create(b, loc, accType, accumulator,
                                     lhs.getMatrix(), rhs.getMatrix());
        mma->setAttr("bosc_ame.ordered", b.getUnitAttr());
        accumulator = mma.getMatrix();
      }
      auto store = TileStoreAccOp::create(b, loc, accumulator, c);
      direct(store, nested(p, "output"), b);
      completion = store;
    }
    return completion;
  };
  auto init = p.getAs<DictionaryAttr>("initialization");
  if (!init)
    return emitTiles(false);
  if (text(init, "kind") == "zero")
    return emitTiles(true);
  // Each arm owns its complete tile lifetime. Matrix values never cross an
  // if-result ABI or a DMA/runtime call, and memory holds every K backedge.
  auto loop = cast<scf::ForOp>(matmul->getParentOp());
  auto first =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                            loop.getInductionVar(), loop.getLowerBound());
  auto branch = scf::IfOp::create(b, loc, first, true);
  b.setInsertionPointToStart(&branch.getThenRegion().front());
  emitTiles(true);
  b.setInsertionPointToStart(&branch.getElseRegion().front());
  emitTiles(false);
  return branch;
}

Operation *emitFill(linalg::FillOp fill) {
  OpBuilder b(fill);
  Location loc = fill.getLoc();
  auto p = plan(fill, "plan.fill");
  Value output = fill.getOutputs()[0], scalar = fill.getInputs()[0];
  auto type = cast<MemRefType>(output.getType());
  if (text(p, "engine") == "ame") {
    // Logical initialization remains a graph task. Its private contents are
    // materialized by the proven first consumer, without zero memory traffic.
    auto deferred = scf::ExecuteRegionOp::create(b, loc, TypeRange{});
    b.setInsertionPointToStart(&deferred.getRegion().emplaceBlock());
    scf::YieldOp::create(b, loc);
    return deferred;
  }
  if (text(p, "level") == "host")
    return memref::StoreOp::create(b, loc, scalar, output, ValueRange{});
  int64_t lanes =
      number(p, "vector_bytes") / elementBytes(type.getElementType());
  Value broadcast = vector::BroadcastOp::create(
      b, loc, VectorType::get({lanes}, type.getElementType()), scalar);
  int64_t rows = number(p, "rows");
  int64_t columns =
      number(p, "row_bytes") / elementBytes(type.getElementType());
  Value zero = index(b, loc, 0);
  auto outer =
      scf::ForOp::create(b, loc, zero, index(b, loc, rows), index(b, loc, 1));
  b.setInsertionPointToStart(outer.getBody());
  auto inner = scf::ForOp::create(b, loc, zero, index(b, loc, columns),
                                  index(b, loc, lanes));
  b.setInsertionPointToStart(inner.getBody());
  SmallVector<Value> indices;
  if (type.getRank() == 2)
    indices.push_back(outer.getInductionVar());
  indices.push_back(inner.getInductionVar());
  vector::StoreOp::create(b, loc, broadcast, output, indices);
  return outer;
}

Operation *emitGeneric(linalg::GenericOp generic) {
  OpBuilder b(generic);
  Location loc = generic.getLoc();
  auto p = plan(generic, "plan.vector");
  int64_t lanes = number(p, "lanes");
  bool scalarMath = text(p, "engine") == "scalar";
  auto loop = scf::ForOp::create(b, loc, index(b, loc, 0),
                                 index(b, loc, number(p, "elements")),
                                 index(b, loc, lanes));
  b.setInsertionPointToStart(loop.getBody());
  Block &body = generic.getRegion().front();
  IRMapping mapping;
  auto selections = p.getAs<ArrayAttr>("operands");
  for (auto [i, argument] : llvm::enumerate(body.getArguments())) {
    if (argument.use_empty())
      continue;
    Value operand = generic->getOperand(i);
    auto type = dyn_cast<MemRefType>(operand.getType());
    if (!type) {
      if (scalarMath) {
        mapping.map(argument, operand);
        continue;
      }
      Value broadcast = vector::BroadcastOp::create(
          b, loc, VectorType::get({lanes}, operand.getType()), operand);
      mapping.map(argument, broadcast);
      continue;
    }
    auto loaded = vector::LoadOp::create(
        b, loc, VectorType::get({lanes}, type.getElementType()), operand,
        ValueRange{loop.getInductionVar()});
    direct(loaded, cast<DictionaryAttr>(selections[i]), b);
    Value value = loaded.getResult();
    if (scalarMath)
      value = vector::ExtractOp::create(b, loc, value, int64_t{0});
    mapping.map(argument, value);
  }
  if (auto scalars = p.getAs<ArrayAttr>("scalar_operands"))
    for (Attribute entry : scalars) {
      auto selection = cast<DictionaryAttr>(entry);
      auto value = cast<TypedAttr>(selection.get("value"));
      Value scalar = arith::ConstantOp::create(b, loc, value.getType(), value);
      Value broadcast = scalar;
      if (!scalarMath)
        broadcast = vector::BroadcastOp::create(
            b, loc, VectorType::get({lanes}, value.getType()), scalar);
      auto operationIndex = selection.getAs<IntegerAttr>("operation");
      Operation &arithmetic = *std::next(
          body.begin(), operationIndex ? operationIndex.getInt() : 0);
      mapping.map(arithmetic.getOperand(number(selection, "operand")),
                  broadcast);
    }
  for (Operation &arithmetic : body.without_terminator()) {
    Type type = arithmetic.getResult(0).getType();
    if (!scalarMath)
      type = VectorType::get({lanes}, type);
    if (isSupportedMathApprox(arithmetic)) {
      mapping.map(arithmetic.getResult(0),
                  emitMathApprox(b, arithmetic,
                                 mapping.lookup(arithmetic.getOperand(0))));
      continue;
    }
    if (isa<math::AbsFOp>(arithmetic)) {
      // abs is an exact sign-bit operation, including NaN payloads and -0.
      // Avoid math-to-libm scalarization and vector spills across fabsf calls.
      Type bitsType = scalarMath
                          ? Type(b.getI32Type())
                          : Type(VectorType::get({lanes}, b.getI32Type()));
      auto bits = arith::BitcastOp::create(
          b, loc, bitsType, mapping.lookup(arithmetic.getOperand(0)));
      Value mask =
          scalarMath
              ? Value(arith::ConstantIntOp::create(b, loc, 0x7fffffff, 32))
              : Value(arith::ConstantOp::create(
                    b, loc,
                    DenseElementsAttr::get(cast<VectorType>(bitsType),
                                           int32_t{0x7fffffff})));
      auto positive = arith::AndIOp::create(b, loc, bits, mask);
      mapping.map(arithmetic.getResult(0),
                  arith::BitcastOp::create(b, loc, type, positive));
      continue;
    }
    if (auto constant = dyn_cast<arith::ConstantOp>(arithmetic)) {
      Value scalar = arith::ConstantOp::create(b, loc, constant.getValue());
      mapping.map(constant.getResult(),
                  scalarMath ? scalar
                             : Value(vector::BroadcastOp::create(b, loc, type,
                                                                 scalar)));
      continue;
    }
    OperationState state(loc, arithmetic.getName());
    for (Value operand : arithmetic.getOperands())
      state.addOperands(mapping.lookup(operand));
    state.addTypes(type);
    state.addAttributes(arithmetic.getAttrs());
    mapping.map(arithmetic.getResult(0), b.create(state)->getResult(0));
  }
  Value result = mapping.lookup(
      cast<linalg::YieldOp>(body.getTerminator()).getValues()[0]);
  if (scalarMath)
    result = vector::BroadcastOp::create(
        b, loc, VectorType::get({1}, result.getType()), result);
  auto stored = vector::StoreOp::create(b, loc, result, generic.getOutputs()[0],
                                        ValueRange{loop.getInductionVar()});
  direct(stored,
         cast<DictionaryAttr>(selections[generic->getNumOperands() - 1]), b);
  return loop;
}

Operation *emitReduce(linalg::ReduceOp reduce) {
  OpBuilder b(reduce);
  Location loc = reduce.getLoc();
  auto p = plan(reduce, "plan.reduce");
  Value source = reduce.getInputs()[0], output = reduce.getInits()[0];
  int64_t lanes = number(p, "vector_bytes") / 4;
  Value initial = memref::LoadOp::create(b, loc, output, ValueRange{});
  auto loop = scf::ForOp::create(b, loc, index(b, loc, 0),
                                 index(b, loc, number(p, "elements")),
                                 index(b, loc, lanes), ValueRange{initial});
  b.setInsertionPointToStart(loop.getBody());
  auto loaded =
      vector::LoadOp::create(b, loc, VectorType::get({lanes}, b.getF32Type()),
                             source, ValueRange{loop.getInductionVar()});
  direct(loaded, nested(p, "input"), b);
  Value next;
  if (text(p, "strategy") != "ordered-scalar-max") {
    auto kind = text(p, "strategy") == "ordered-vector-maxnum"
                    ? vector::CombiningKind::MAXNUMF
                    : vector::CombiningKind::ADD;
    next = vector::ReductionOp::create(b, loc, kind, loaded,
                                       loop.getRegionIterArgs()[0]);
  } else {
    // Preserve maximumf NaN and signed-zero semantics, as well as the
    // original operand order. The vector port supplies one SPM element at a
    // time; the scalar combiner does not introduce a library call.
    Value element =
        vector::ExtractOp::create(b, loc, loaded, ArrayRef<int64_t>{0});
    Block &body = reduce.getRegion().front();
    IRMapping mapping;
    mapping.map(body.getArgument(0), element);
    mapping.map(body.getArgument(1), loop.getRegionIterArgs()[0]);
    next = b.clone(body.front(), mapping)->getResult(0);
  }
  scf::YieldOp::create(b, loc, next);
  b.setInsertionPointAfter(loop);
  return memref::StoreOp::create(b, loc, loop.getResult(0), output,
                                 ValueRange{});
}

Operation *emitScalarLoad(memref::LoadOp load) {
  OpBuilder b(load);
  auto type = cast<MemRefType>(load.getMemRef().getType());
  auto vector = vector::LoadOp::create(
      b, load.getLoc(), VectorType::get({1}, type.getElementType()),
      load.getMemRef(), load.getIndices());
  auto extracted =
      vector::ExtractOp::create(b, load.getLoc(), vector, int64_t{0});
  load.getResult().replaceAllUsesWith(extracted);
  return extracted;
}

struct EmitPass : PassWrapper<EmitPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitPass)
  StringRef getArgument() const final { return "bosc-ame-emit"; }
  StringRef getDescription() const final {
    return "Mechanically expand verified BOSCAME plans";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry
        .insert<BOSCAMEDialect, arith::ArithDialect, func::FuncDialect,
                linalg::LinalgDialect, math::MathDialect, memref::MemRefDialect,
                scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() final {
    if (failed(emitBOSCAMEPlan(getOperation())))
      signalPassFailure();
  }
};
} // namespace

LogicalResult buddy::boscame::emitBOSCAMEPlan(ModuleOp module) {
  // No mutation precedes full validation. No legacy planner is called here.
  if (failed(verifyBOSCAMEPlan(module)))
    return failure();
  DenseMap<Operation *, DenseSet<int64_t>> expected;
  module.walk([&](Operation *op) {
    if (isStreamMember(op) || isRegisterScheduleMember(op) ||
        isFloatScheduleMember(op))
      return;
    if (auto task = plan(op, "contract.task"))
      expected[op->getParentOfType<func::FuncOp>()].insert(number(task, "id"));
  });
  OpBuilder builder(module.getContext());
  module->setAttr("bosc_ame.target", builder.getStringAttr("nr-fpga"));
  bool streamingPlan = false;
  if (auto version = module->getAttrOfType<IntegerAttr>("plan.version"))
    streamingPlan = version.getInt() == 2;
  for (auto function : module.getOps<func::FuncOp>()) {
    if (streamingPlan) {
      if (Attribute graph = function->getAttr("contract.compute_graph")) {
        function->setAttr("bosc_ame.logical_compute_graph", graph);
        function->removeAttr("contract.compute_graph");
      }
      function->setAttr("bosc_ame.task_graph_semantics",
                        builder.getStringAttr("logical-provenance"));
    }
    if (Attribute mapping = function->getAttr("plan.task_map"))
      function->setAttr("bosc_ame.task_map", mapping);
    if (Attribute count = function->getAttr("plan.source_task_count"))
      function->setAttr("bosc_ame.source_task_count", count);
  }
  auto abi = plan(module, "contract.abi");
  if (abi && abi.getAs<IntegerAttr>("args_aligned_bytes") &&
      number(abi, "args_aligned_bytes") >= 64 &&
      number(abi, "args_aligned_bytes") % 64 == 0) {
    auto disjoint = abi.getAs<BoolAttr>("args_disjoint");
    module->setAttr(
        spmdma::kDmaAbiAttrName,
        builder.getStringAttr(disjoint && disjoint.getValue()
                                  ? spmdma::kDmaAbiAlignedNonAliasing
                                  : spmdma::kDmaAbiAligned));
  }
  if (failed(emitRegisterSchedules(module)) ||
      failed(emitFloatSchedules(module)) || failed(emitStreams(module, true)))
    return failure();
  materializeBuffers(module);
  SmallVector<Operation *> computations;
  module.walk([&](Operation *op) {
    if (isa<memref::CopyOp, linalg::MatmulOp, linalg::FillOp, linalg::GenericOp,
            linalg::ReduceOp>(op) ||
        op->hasAttr("plan.scalar") || op->hasAttr("plan.cpu"))
      computations.push_back(op);
  });
  for (Operation *op : computations) {
    Operation *completion = nullptr;
    if (op->hasAttr("plan.cpu"))
      completion = emitDDRRecipe(op);
    else if (auto copy = dyn_cast<memref::CopyOp>(op))
      completion = emitTransfer(copy);
    else if (auto matmul = dyn_cast<linalg::MatmulOp>(op))
      completion = matmul->hasAttr("plan.compute") ? emitOrderedMatmul(matmul)
                                                   : emitMatmul(matmul);
    else if (auto fill = dyn_cast<linalg::FillOp>(op))
      completion = emitFill(fill);
    else if (auto generic = dyn_cast<linalg::GenericOp>(op))
      completion = emitGeneric(generic);
    else if (auto reduce = dyn_cast<linalg::ReduceOp>(op))
      completion = emitReduce(reduce);
    else if (auto load = dyn_cast<memref::LoadOp>(op))
      completion = emitScalarLoad(load);
    if (!completion)
      return op->emitError(
          "verified plan did not produce a completion operation");
    moveFacts(op, completion);
    op->erase();
  }
  bool invalid = false;
  DenseMap<Operation *, DenseSet<int64_t>> actual;
  module.walk([&](Operation *op) {
    if (auto task = plan(op, "contract.task"))
      if (!actual[op->getParentOfType<func::FuncOp>()]
               .insert(number(task, "id"))
               .second) {
        op->emitError("duplicate task completion after planned emission");
        invalid = true;
      }
    SmallVector<StringAttr> consumed;
    for (NamedAttribute attr : op->getAttrs())
      if (attr.getName().getValue().starts_with("plan."))
        consumed.push_back(attr.getName());
    for (StringAttr name : consumed)
      op->removeAttr(name);
  });
  for (auto &[function, ids] : expected)
    if (actual[function].size() != ids.size() ||
        llvm::any_of(
            ids, [&](int64_t id) { return !actual[function].contains(id); })) {
      function->emitError(
          "planned task completion IDs were lost during emission");
      invalid = true;
    }
  if (invalid)
    return failure();
  return verify(module);
}
std::unique_ptr<Pass> buddy::boscame::createBOSCAMEEmitPass() {
  return std::make_unique<EmitPass>();
}
void buddy::boscame::registerBOSCAMEPlanEmissionPasses() {
  registerPass([] { return createBOSCAMEPlanVerifyPass(); });
  registerPass([] { return createBOSCAMEEmitPass(); });
}
