//===- LowerLinalgToBOSCAME.cpp - Compatibility input adapters ------------===//
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
// Compatibility ModuleOp pass that validates neutral buffer-form input,
// normalizes supported generics, rebuilds contract metadata when needed, then
// runs the shared BOSCAME plan and emit pipeline.
//
//===----------------------------------------------------------------------===//

#include "Conversion/LowerLinalgToBOSCAME/LowerLinalgToBOSCAME.h"
#include "Analysis/KernelContract.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEBufferNormalization.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEPlanIR.h"
#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

using namespace mlir;
using namespace ::buddy::boscame;

namespace {
Value rootOf(Value value) {
  while (auto view =
             dyn_cast_or_null<ViewLikeOpInterface>(value.getDefiningOp()))
    value = view.getViewSource();
  return value;
}

MemRefType neutralType(MemRefType type) {
  return MemRefType::get(type.getShape(), type.getElementType(),
                         type.getLayout());
}

LogicalResult validateCompatibilityDialects(ModuleOp module) {
  return failure(
      module
          .walk([&](Operation *op) -> WalkResult {
            StringRef dialect = op->getName().getDialectNamespace();
            if (op->getName().getStringRef() !=
                    "builtin.unrealized_conversion_cast" &&
                llvm::is_contained(
                    ArrayRef<StringRef>{"builtin", "func", "arith", "affine",
                                        "scf", "cf", "math", "memref", "linalg",
                                        "vector", "bosc_ame"},
                    dialect))
              return WalkResult::advance();
            op->emitError("compatibility input contains a non-neutral dialect "
                          "or bridge operation; lower the frontend before "
                          "BOSCAME planning");
            return WalkResult::interrupt();
          })
          .wasInterrupted());
}

DictionaryAttr boundary(OpBuilder &builder, StringRef kind,
                        DictionaryAttr legacy = {}) {
  NamedAttrList fields;
  fields.append("kind", builder.getStringAttr(kind));
  fields.append("boundary", builder.getStringAttr("external_ddr"));
  NamedAttrList mask;
  auto masked = legacy ? legacy.getAs<BoolAttr>("masked") : BoolAttr();
  mask.append("kind",
              builder.getStringAttr(masked && masked.getValue() ? "rectangular"
                                                                : "none"));
  if (masked && masked.getValue())
    if (auto valid = legacy.get("valid_shape"))
      mask.append("valid_shape", valid);
  fields.append("mask", builder.getDictionaryAttr(mask));
  if (legacy)
    for (StringRef key : {"origin_id", "other"})
      if (auto attr = legacy.get(key))
        fields.append(key, attr);
  return builder.getDictionaryAttr(fields);
}

// A physical window can include untouched guard bytes. Extract one independent
// logical subwindow rather than claiming those bytes were initialized by a
// partial load. Every alias and actual memory access must be statically inside
// the extracted interval, and the original pointer/extent must be
// unobservable. This changes only descriptors and their fixed placement; no
// bytes are moved.
void extractLogicalSubwindow(FixedSPMBuffer &fixed) {
  auto rootType = dyn_cast<MemRefType>(fixed.root.getType());
  if (!rootType || rootType.getRank() != 1)
    return;
  int64_t width = spmdma::getSpmElementByteWidth(rootType.getElementType());
  if (!width)
    return;
  auto range = [&](Value value) -> std::optional<std::pair<int64_t, int64_t>> {
    auto type = dyn_cast<MemRefType>(value.getType());
    auto offset = spmdma::getStaticDescriptorOffset(value);
    SmallVector<int64_t> strides;
    int64_t ignored, end;
    if (!type || !type.hasStaticShape() || !offset || *offset < 0 ||
        type.getElementType() != rootType.getElementType() ||
        failed(type.getStridesAndOffset(strides, ignored)))
      return std::nullopt;
    end = *offset;
    for (auto [dimension, stride] : llvm::zip(type.getShape(), strides)) {
      int64_t delta;
      if (dimension <= 0 || stride <= 0 ||
          llvm::MulOverflow(dimension - 1, stride, delta) ||
          llvm::AddOverflow(end, delta, end))
        return std::nullopt;
    }
    if (llvm::AddOverflow(end, int64_t{1}, end))
      return std::nullopt;
    return std::pair(*offset, end);
  };
  SmallVector<Value> work{fixed.root}, aliases;
  llvm::DenseSet<Value> visited;
  int64_t first = std::numeric_limits<int64_t>::max(), last = 0;
  while (!work.empty()) {
    Value value = work.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (Operation *user : value.getUsers()) {
      if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
        if (view.getViewSource() != value ||
            !isa<memref::SubViewOp, memref::ReinterpretCastOp,
                 memref::CollapseShapeOp, memref::ExpandShapeOp,
                 memref::TransposeOp>(user))
          return;
        if (auto sub = dyn_cast<memref::SubViewOp>(user);
            sub && value == fixed.root &&
            (sub.getStaticOffsets().size() != 1 ||
             ShapedType::isDynamic(sub.getStaticOffsets()[0])))
          return;
        if (auto cast = dyn_cast<memref::ReinterpretCastOp>(user);
            cast && ShapedType::isDynamic(cast.getStaticOffsets()[0]))
          return;
        for (Value alias : user->getResults()) {
          if (!isa<MemRefType>(alias.getType()))
            return;
          aliases.push_back(alias);
          work.push_back(alias);
        }
        continue;
      }
      if (isa<memref::DeallocOp>(user))
        continue;
      if (!isa<memref::CopyOp, memref::LoadOp, memref::StoreOp, vector::LoadOp,
               vector::StoreOp, linalg::FillOp, linalg::GenericOp,
               linalg::MatmulOp, linalg::ReduceOp>(user))
        return;
      auto bounds = range(value);
      if (!bounds)
        return;
      first = std::min(first, bounds->first);
      last = std::max(last, bounds->second);
    }
  }
  int64_t firstBytes, lastBytes;
  if (first == std::numeric_limits<int64_t>::max() || first < 0 ||
      llvm::MulOverflow(first, width, firstBytes) ||
      llvm::MulOverflow(last, width, lastBytes) || firstBytes % 64 ||
      lastBytes % 64 || lastBytes > fixed.capacity)
    return;
  if (firstBytes == 0 && lastBytes == fixed.capacity)
    return;
  SmallVector<int64_t> aliasOffsets;
  for (Value alias : aliases) {
    auto bounds = range(alias);
    if (!bounds || bounds->first < first || bounds->second > last)
      return;
    aliasOffsets.push_back(bounds->first);
  }
  // All checks finish before modifying any descriptor.
  for (auto [alias, absoluteOffset] : llvm::zip(aliases, aliasOffsets)) {
    Operation *op = alias.getDefiningOp();
    if (auto sub = dyn_cast<memref::SubViewOp>(op);
        sub && sub.getSource() == fixed.root)
      sub.setStaticOffsets(
          ArrayRef<int64_t>{sub.getStaticOffsets()[0] - first});
    if (auto cast = dyn_cast<memref::ReinterpretCastOp>(op))
      cast.setStaticOffsets(
          ArrayRef<int64_t>{cast.getStaticOffsets()[0] - first});
    auto type = cast<MemRefType>(alias.getType());
    SmallVector<int64_t> strides;
    int64_t ignored;
    (void)type.getStridesAndOffset(strides, ignored);
    alias.setType(MemRefType::get(type.getShape(), type.getElementType(),
                                  StridedLayoutAttr::get(type.getContext(),
                                                         absoluteOffset - first,
                                                         strides)));
  }
  fixed.root.setType(
      MemRefType::get({last - first}, rootType.getElementType()));
  fixed.offset += firstBytes;
  fixed.capacity = lastBytes - firstBytes;
}

// Translate only facts whose meaning is unchanged. Target configuration is
// validated separately; unrecognized target metadata is diagnosed rather
// than being silently discarded by contract annotation.
LogicalResult migrateSourceFacts(ModuleOp module) {
  OpBuilder builder(module.getContext());
  return failure(
      module
          .walk([&](Operation *op) -> WalkResult {
            if (auto function = dyn_cast<func::FuncOp>(op)) {
              for (unsigned index = 0; index != function.getNumArguments();
                   ++index) {
                auto role = function.getArgAttrOfType<StringAttr>(
                    index, "bosc_ame.program_role");
                auto axis = function.getArgAttrOfType<IntegerAttr>(
                    index, "bosc_ame.program_axis");
                if (role || axis) {
                  if (!role || !axis || axis.getInt() < 0 ||
                      axis.getInt() > 2 ||
                      (role.getValue() != "program_id" &&
                       role.getValue() != "num_programs")) {
                    function.emitError("invalid legacy program-axis metadata");
                    return WalkResult::interrupt();
                  }
                  function.setArgAttr(
                      index, "contract.program",
                      builder.getDictionaryAttr(
                          {builder.getNamedAttr("role", role),
                           builder.getNamedAttr("axis", axis)}));
                  function.removeArgAttr(index, "bosc_ame.program_role");
                  function.removeArgAttr(index, "bosc_ame.program_axis");
                }
                SmallVector<StringAttr> derived;
                auto argumentAttrs = function.getArgAttrDict(index);
                for (NamedAttribute attr : argumentAttrs
                                               ? argumentAttrs.getValue()
                                               : ArrayRef<NamedAttribute>{}) {
                  StringRef name = attr.getName().strref();
                  if (!name.starts_with("bosc_ame."))
                    continue;
                  StringRef key = name.drop_front(9);
                  if (key == "logical_value_id" || key == "logical_value_ids" ||
                      key == "logical_value_users" ||
                      key == "varying_program_axes" ||
                      key == "invariant_program_axes") {
                    derived.push_back(attr.getName());
                    continue;
                  }
                  function.emitError()
                      << "unsupported legacy target argument metadata: " << name
                      << "; regenerate a neutral contract";
                  return WalkResult::interrupt();
                }
                for (StringAttr key : derived)
                  function.removeArgAttr(index, key);
              }
              if (auto grid = function->getAttr("gpu.known_grid_size")) {
                auto extents = dyn_cast<DenseI32ArrayAttr>(grid);
                if (!extents) {
                  function.emitError(
                      "gpu.known_grid_size must be a dense i32 array");
                  return WalkResult::interrupt();
                }
                if (extents.size() != 3) {
                  function.emitError(
                      "gpu.known_grid_size must contain exactly 3 elements");
                  return WalkResult::interrupt();
                }
                if (llvm::any_of(extents.asArrayRef(),
                                 [](int32_t extent) { return extent <= 0; })) {
                  function.emitError("invalid gpu.known_grid_size contract; "
                                     "expected three positive i32 extents");
                  return WalkResult::interrupt();
                }
                function->setAttr("contract.grid", grid);
              }
            }
            NamedAttrList provenance;
            if (auto old =
                    op->getAttrOfType<DictionaryAttr>("contract.provenance"))
              provenance.assign(old.getValue());
            SmallVector<StringAttr> remove;
            for (NamedAttribute attr : op->getAttrs()) {
              StringRef name = attr.getName().strref();
              if (!name.starts_with("bosc_ame."))
                continue;
              StringRef key = name.drop_front(9);
              if (key == "memory") {
                auto memory = dyn_cast<DictionaryAttr>(attr.getValue());
                auto kind =
                    memory ? memory.getAs<StringAttr>("kind") : StringAttr();
                if (!isa<memref::CopyOp>(op) || !kind ||
                    (kind.getValue() != "load" && kind.getValue() != "store")) {
                  op->emitError(
                      "legacy memory facts require a concrete buffer copy");
                  return WalkResult::interrupt();
                }
                op->setAttr("contract.access",
                            boundary(builder, kind.getValue(), memory));
              } else if (key == "padding") {
                if (!isa<linalg::FillOp>(op)) {
                  op->emitError("padding metadata requires a concrete fill");
                  return WalkResult::interrupt();
                }
                op->setAttr("contract.fill",
                            builder.getDictionaryAttr({builder.getNamedAttr(
                                "padding", builder.getBoolAttr(true))}));
              } else if (key == "origin_id" || key == "compute_origin_id") {
                provenance.set("origin_id", attr.getValue());
              } else if (key.ends_with("_origin_id") ||
                         key.ends_with("_origin_ids") ||
                         key.ends_with("_lineage_complete")) {
                provenance.set(key, attr.getValue());
              } else if (key == "lower_to_ame") {
                auto matmul = dyn_cast<linalg::MatmulOp>(op);
                if (!matmul ||
                    matmul.getCast() != linalg::TypeFn::cast_signed ||
                    !cast<ShapedType>(matmul.getInputs()[0].getType())
                         .getElementType()
                         .isInteger(8) ||
                    !cast<ShapedType>(matmul.getInputs()[1].getType())
                         .getElementType()
                         .isInteger(8) ||
                    !cast<ShapedType>(matmul.getOutputs()[0].getType())
                         .getElementType()
                         .isInteger(32)) {
                  op->emitError("legacy AME marker requires signed i8 x i8 -> "
                                "i32 matmul");
                  return WalkResult::interrupt();
                }
              } else if (key.starts_with("compute_") ||
                         key.starts_with("logical_value") ||
                         key == "varying_program_axes" ||
                         key == "invariant_program_axes" ||
                         key == "rematerializable") {
                // These derived facts are rebuilt from the actual standard
                // operations.
              } else {
                op->emitError()
                    << "unsupported legacy target metadata: " << name
                    << "; remove it and regenerate a neutral contract";
                return WalkResult::interrupt();
              }
              remove.push_back(attr.getName());
            }
            for (StringAttr key : remove)
              op->removeAttr(key);
            if (!provenance.empty())
              op->setAttr("contract.provenance",
                          builder.getDictionaryAttr(provenance));
            if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
              bool ame = matmul.getCast() == linalg::TypeFn::cast_signed &&
                         cast<ShapedType>(matmul.getInputs()[0].getType())
                             .getElementType()
                             .isInteger(8) &&
                         cast<ShapedType>(matmul.getInputs()[1].getType())
                             .getElementType()
                             .isInteger(8) &&
                         cast<ShapedType>(matmul.getOutputs()[0].getType())
                             .getElementType()
                             .isInteger(32);
              if (!ame && !op->hasAttr("contract.compute"))
                op->setAttr("contract.compute",
                            builder.getDictionaryAttr(
                                {builder.getNamedAttr(
                                     "kind", builder.getStringAttr("matmul")),
                                 builder.getNamedAttr(
                                     "order", builder.getStringAttr("mnk"))}));
            }
            return WalkResult::advance();
          })
          .wasInterrupted());
}

// Physical compatibility is deliberately outside the neutral interchange.
// Validate original windows and synchronous transfer order before erasing
// target types. Pass their placement constraints to the one planner in C++.
LogicalResult adaptPhysicalInput(ModuleOp module,
                                 SmallVectorImpl<FixedSPMBuffer> &fixed) {
  SmallVector<SpmViewOp> windows;
  SmallVector<DmaStartOp> transfers;
  WalkResult surface = module.walk([&](Operation *op) -> WalkResult {
    if (auto window = dyn_cast<SpmViewOp>(op)) {
      if (failed(window.verify()))
        return WalkResult::interrupt();
      windows.push_back(window);
    } else if (auto transfer = dyn_cast<DmaStartOp>(op)) {
      if (failed(transfer.verify()))
        return WalkResult::interrupt();
      transfers.push_back(transfer);
    } else if (op->getName().getDialectNamespace() == "bosc_ame" &&
               !isa<DmaWaitOp>(op)) {
      op->emitError(
          "target compatibility accepts SPM windows and synchronous DMA only; "
          "already emitted instructions cannot be replanned");
      return WalkResult::interrupt();
    }
    auto validate = [&](Value value) {
      auto type = dyn_cast<BaseMemRefType>(value.getType());
      if (!type || !type.getMemorySpace())
        return success();
      auto bank = spmdma::getSpmBank(type);
      if (!bank || !rootOf(value).getDefiningOp<SpmViewOp>())
        return op->emitError(
                   "target memory must be rooted in a verified SPM window"),
               failure();
      auto viewType = dyn_cast<MemRefType>(type);
      auto origin = spmdma::getStaticPointerOrigin(value);
      auto offset = spmdma::getStaticDescriptorOffset(value);
      SmallVector<int64_t> strides;
      int64_t ignored = 0;
      if (!viewType || !viewType.hasStaticShape() || !origin || !offset ||
          *offset < 0 || origin->byteShift < 0 ||
          failed(viewType.getStridesAndOffset(strides, ignored)))
        return op->emitError("SPM input view needs a known nonnegative "
                             "descriptor offset and static layout"),
               failure();
      int64_t end = *offset;
      for (auto [dimension, stride] : llvm::zip(viewType.getShape(), strides)) {
        int64_t delta = 0;
        if (dimension <= 0 || stride <= 0 ||
            llvm::MulOverflow(dimension - 1, stride, delta) ||
            llvm::AddOverflow(end, delta, end))
          return op->emitError(
                     "SPM input view has an unbounded physical range"),
                 failure();
      }
      int64_t width = spmdma::getSpmElementByteWidth(viewType.getElementType());
      auto capacity = spmdma::getContiguousZeroOffsetByteSize(
          cast<MemRefType>(rootOf(value).getType()));
      if (!width || !capacity || llvm::AddOverflow(end, int64_t{1}, end) ||
          llvm::MulOverflow(end, width, end) ||
          llvm::AddOverflow(end, origin->byteShift, end) || end > *capacity)
        return op->emitError("SPM input view leaves its spm.view allocation"),
               failure();
      return success();
    };
    for (Value value : op->getOperands())
      if (failed(validate(value)))
        return WalkResult::interrupt();
    for (Value value : op->getResults())
      if (failed(validate(value)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (surface.wasInterrupted())
    return failure();
  // A copy has synchronous completion. Only erase DMA when that completion
  // has exactly the same observable position and exact endpoint byte range.
  for (DmaStartOp start : transfers) {
    if (!start.getToken().hasOneUse())
      return start.emitError(
          "target DMA adapter requires one local completion");
    auto wait = dyn_cast<DmaWaitOp>(*start.getToken().getUsers().begin());
    if (!wait || wait->getBlock() != start->getBlock() ||
        !start->isBeforeInBlock(wait))
      return start.emitError(
          "target DMA adapter requires a following local completion");
    for (Operation *between = start->getNextNode(); between != wait;
         between = between->getNextNode())
      if (!isMemoryEffectFree(between) || between->getNumRegions())
        return between->emitError("cannot normalize asynchronous DMA across an "
                                  "intervening effect or region");
    APInt bytes;
    auto src = cast<MemRefType>(start.getSrc().getType());
    auto dst = cast<MemRefType>(start.getDst().getType());
    if (!matchPattern(start.getBytes(), m_ConstantInt(&bytes)) ||
        src.getElementType() != dst.getElementType() ||
        spmdma::getSpmBank(src).has_value() ==
            spmdma::getSpmBank(dst).has_value())
      return start.emitError(
          "target DMA adapter requires a static typed DDR/SPM transfer");
    int64_t elementBytes = spmdma::getSpmElementByteWidth(src.getElementType());
    int64_t elements = bytes.getSExtValue() / elementBytes;
    OpBuilder builder(wait);
    auto prefix = [&](Value endpoint) -> Value {
      auto type = cast<MemRefType>(endpoint.getType());
      if (elements == type.getDimSize(0))
        return endpoint;
      return memref::SubViewOp::create(
          builder, wait.getLoc(), endpoint,
          ArrayRef<OpFoldResult>{builder.getIndexAttr(0)},
          ArrayRef<OpFoldResult>{builder.getIndexAttr(elements)},
          ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
    };
    auto copy = memref::CopyOp::create(
        builder, wait.getLoc(), prefix(start.getSrc()), prefix(start.getDst()));
    copy->setAttr(
        "contract.access",
        boundary(builder, spmdma::getSpmBank(src) ? "store" : "load"));
    for (NamedAttribute attr : wait->getAttrs())
      if (attr.getName().strref().starts_with("contract."))
        copy->setAttr(attr.getName(), attr.getValue());
    wait.erase();
    start.erase();
  }
  // Check sharing after synchronous DMA becomes concrete copies, so the same
  // arm-local noescape/completion proof is used for raw and planned inputs.
  DominanceInfo dominance(module);
  llvm::DenseSet<Operation *> duplicates;
  for (auto [index, current] : llvm::enumerate(windows)) {
    auto capacity = spmdma::getContiguousZeroOffsetByteSize(
        cast<MemRefType>(current.getView().getType()));
    if (!capacity)
      return current.emitError("SPM input needs a bounded contiguous root");
    for (SpmViewOp previous : ArrayRef(windows).take_front(index)) {
      if (duplicates.contains(previous) ||
          current->getParentOfType<func::FuncOp>() !=
              previous->getParentOfType<func::FuncOp>() ||
          current.getBank() != previous.getBank() ||
          canShareExclusiveSPMWindow(current.getView(), previous.getView()))
        continue;
      auto previousCapacity = *spmdma::getContiguousZeroOffsetByteSize(
          cast<MemRefType>(previous.getView().getType()));
      if (current.getOffset() >= previous.getOffset() + previousCapacity ||
          previous.getOffset() >= current.getOffset() + *capacity)
        continue;
      if (current.getOffset() == previous.getOffset() &&
          current.getView().getType() == previous.getView().getType() &&
          dominance.dominates(previous.getOperation(),
                              current.getOperation())) {
        current.getView().replaceAllUsesWith(previous.getView());
        duplicates.insert(current);
        break;
      }
      return current.emitError(
          "overlapping SPM input windows require a shared root or provably "
          "exclusive arms; distinct private allocations would change aliasing");
    }
  }
  for (SpmViewOp window : windows) {
    if (duplicates.contains(window)) {
      window.erase();
      continue;
    }
    auto type = cast<MemRefType>(window.getView().getType());
    uint64_t capacity = *spmdma::getContiguousZeroOffsetByteSize(type);
    if (window.getOffset() > uint64_t(std::numeric_limits<int64_t>::max()) ||
        capacity > uint64_t(std::numeric_limits<int64_t>::max()))
      return window.emitError("SPM input offset and capacity must fit signed "
                              "64-bit planning fields");
    OpBuilder builder(window);
    auto root =
        memref::AllocOp::create(builder, window.getLoc(), neutralType(type),
                                builder.getI64IntegerAttr(64));
    fixed.push_back({root, window.getBank(), int64_t(window.getOffset()),
                     int64_t(capacity)});
    // Rewrite users first; their view types are normalized in the next walk.
    window.getView().replaceAllUsesWith(root);
    window.erase();
  }
  module.walk([&](Operation *op) {
    for (Value result : op->getResults())
      if (auto type = dyn_cast<MemRefType>(result.getType());
          type && spmdma::getSpmBank(type))
        result.setType(neutralType(type));
  });
  for (FixedSPMBuffer &window : fixed)
    extractLogicalSubwindow(window);
  return success();
}

// The two historical names deliberately share every input and output step.
template <bool Independent>
class LowerLinalgToBOSCAMEPass
    : public PassWrapper<LowerLinalgToBOSCAMEPass<Independent>,
                         OperationPass<ModuleOp>> {
public:
  using Base = PassWrapper<LowerLinalgToBOSCAMEPass<Independent>,
                           OperationPass<ModuleOp>>;
  template <typename T> using Option = typename Base::template Option<T>;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToBOSCAMEPass)
  LowerLinalgToBOSCAMEPass() = default;
  LowerLinalgToBOSCAMEPass(const LowerLinalgToBOSCAMEPass &) {}
  StringRef getArgument() const final {
    return Independent ? "linalg-to-boscame" : "lower-linalg-to-boscame";
  }
  StringRef getDescription() const final {
    return "Adapt buffer-form input through the shared BOSCAME contract, "
           "planner and emitter";
  }
  Option<bool> enableSpmDma{
      *this, "enable-spm-dma",
      llvm::cl::desc("Compatibility assertion that SPM/DMA is available"),
      llvm::cl::init(true)};
  Option<bool> enableAME{
      *this, "enable-ame",
      llvm::cl::desc("Compatibility assertion that AME is available"),
      llvm::cl::init(true)};
  Option<std::string> dmaAbi{
      *this, "dma-abi",
      llvm::cl::desc("Explicit caller promise: args_aligned64 or "
                     "args_aligned64_nonalias"),
      llvm::cl::init("")};
  Option<std::string> target{*this, "target",
                             llvm::cl::desc("Target profile: nr-fpga"),
                             llvm::cl::init("nr-fpga")};
  Option<int64_t> bm{*this, "bm", llvm::cl::desc("Obsolete geometry override"),
                     llvm::cl::init(64)};
  Option<int64_t> bn{*this, "bn", llvm::cl::desc("Obsolete geometry override"),
                     llvm::cl::init(64)};
  Option<int64_t> bk{*this, "bk", llvm::cl::desc("Obsolete geometry override"),
                     llvm::cl::init(128)};
  Option<bool> bufferPlacement{
      *this, "buffer-placement",
      llvm::cl::desc("Enable optional automatic SPM placement (V1: required "
                     "fixed placement)"),
      llvm::cl::init(true)};
  Option<int64_t> planVersion{
      *this, "plan-version",
      llvm::cl::desc(
          "Plan version (1: frozen compatibility, 2: DDR/register default)"),
      llvm::cl::init(2)};
  Option<bool> planReport{*this, "plan-report",
                          llvm::cl::desc("Print the concrete plan"),
                          llvm::cl::init(false)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<BOSCAMEDialect, arith::ArithDialect, func::FuncDialect,
                linalg::LinalgDialect, math::MathDialect, memref::MemRefDialect,
                scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = this->getOperation();
    auto run = [&]() -> LogicalResult {
      if (module->hasAttr("bosc_ame.store_policy"))
        return module.emitError("bosc_ame.store_policy is obsolete; remove it "
                                "and rerun the automatic BOSCAME planner");
      if (bm.getNumOccurrences() || bn.getNumOccurrences() ||
          bk.getNumOccurrences())
        return module.emitError(
            "legacy bm/bn/bk overrides are obsolete; geometry is taken from "
            "buffer-form Linalg operands");
      if (!enableSpmDma || !enableAME)
        return module.emitError(
            "the automatic BOSCAME planner requires enabled target engines; "
            "remove disabled legacy feature "
            "overrides");
      if (target != "nr-fpga")
        return module.emitError("unsupported BOSCAME target; expected nr-fpga");
      for (StringRef name : {"bosc_ame.enable_spm_dma", "bosc_ame.enable_ame"})
        if (Attribute attr = module->getAttr(name)) {
          auto enabled = dyn_cast<BoolAttr>(attr);
          if (!enabled || !enabled.getValue())
            return module.emitError()
                   << "disabled or invalid legacy target feature " << name
                   << "; remove it and use automatic planning";
          module->removeAttr(name);
        }
      for (StringRef name : {"bosc_ame.bm", "bosc_ame.bn", "bosc_ame.bk"})
        if (module->hasAttr(name))
          return module.emitError()
                 << "obsolete geometry attribute " << name
                 << "; remove it and regenerate a neutral contract";
      if (Attribute attr = module->getAttr("bosc_ame.target")) {
        auto profile = dyn_cast<StringAttr>(attr);
        if (!profile || profile.getValue() != target)
          return module.emitError(
              "invalid or conflicting BOSCAME target profile");
        module->removeAttr("bosc_ame.target");
      }
      StringRef abi = dmaAbi;
      if (Attribute attr = module->getAttr(spmdma::kDmaAbiAttrName)) {
        auto promised = dyn_cast<StringAttr>(attr);
        if (!promised || (!abi.empty() && promised.getValue() != abi))
          return module.emitError(
              "invalid or conflicting explicit DMA ABI contract");
        abi = promised.getValue();
      }
      if (!abi.empty()) {
        if (abi != spmdma::kDmaAbiAlignedNonAliasing &&
            abi != spmdma::kDmaAbiAligned)
          return module.emitError() << "invalid dma-abi: " << abi;
        OpBuilder builder(module.getContext());
        NamedAttrList promises;
        promises.append("args_aligned_bytes", builder.getI64IntegerAttr(64));
        if (abi == spmdma::kDmaAbiAlignedNonAliasing)
          promises.append("args_disjoint", builder.getBoolAttr(true));
        auto promised = promises.getDictionary(module.getContext());
        if (auto existing = module->getAttr("contract.abi");
            existing && existing != promised)
          return module.emitError(
              "explicit DMA ABI conflicts with contract.abi");
        module->setAttr("contract.abi", promised);
        // The target op verifier needs this explicit promise before adaptation.
        module->setAttr(spmdma::kDmaAbiAttrName, builder.getStringAttr(abi));
      }
      if (failed(validateCompatibilityDialects(module)))
        return failure();
      SmallVector<FixedSPMBuffer> fixed;
      if (!module->hasAttr("contract.version")) {
        if (failed(adaptPhysicalInput(module, fixed)))
          return failure();
        module->removeAttr(spmdma::kDmaAbiAttrName);
        if (auto placement = module->getAttr("bosc_ame.placement")) {
          if (placement != StringAttr::get(module.getContext(), "logical"))
            return module.emitError(
                "unsupported legacy placement mode; use automatic planning");
          module->removeAttr("bosc_ame.placement");
        }
        if (auto expected = module->getAttr("bosc_ame.expected_ame_matmuls")) {
          auto count = dyn_cast<IntegerAttr>(expected);
          int64_t actual = 0;
          module.walk([&](linalg::MatmulOp op) {
            actual += op->hasAttr("bosc_ame.lower_to_ame");
          });
          if (!count || count.getInt() != actual)
            return module.emitError(
                "BOSCAME AME marker count changed before input adaptation");
          module->removeAttr("bosc_ame.expected_ame_matmuls");
        }
        normalizeBOSCAMEBufferInput(module);
        if (failed(migrateSourceFacts(module)))
          return failure();
        if (failed(::buddy::contract::annotateKernelContract(module)))
          return failure();
      } else if (!abi.empty()) {
        module->removeAttr(spmdma::kDmaAbiAttrName);
      }
      if (!module->hasAttr("plan.version") &&
          failed(planBOSCAME(module, fixed,
                             PlanOptions{bool(bufferPlacement), planVersion})))
        return failure();
      if (failed(verifyBOSCAMEPlan(module)))
        return failure();
      if (planReport)
        printBOSCAMEPlan(module, llvm::errs());
      return emitBOSCAMEPlan(module);
    };
    if (failed(run()))
      this->signalPassFailure();
  }
};
} // namespace

namespace mlir::buddy {
std::unique_ptr<Pass> createLowerLinalgToBOSCAMEPass() {
  return std::make_unique<LowerLinalgToBOSCAMEPass<false>>();
}
std::unique_ptr<Pass> createLinalgToBOSCAMEPass() {
  return std::make_unique<LowerLinalgToBOSCAMEPass<true>>();
}
void registerLowerLinalgToBOSCAMEPass() {
  PassRegistration<LowerLinalgToBOSCAMEPass<false>>();
  PassRegistration<LowerLinalgToBOSCAMEPass<true>>();
}
} // namespace mlir::buddy
