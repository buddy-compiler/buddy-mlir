//===- BOSCAMEPlanVerify.cpp - Verify concrete BOSCAME access plans -------===//
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
// Re-prove recorded BOSCAME plan attributes against ordinary IR and target
// contracts before any emission mutates the module.
//
//===----------------------------------------------------------------------===//

#include "Analysis/KernelContract.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEAccumulatorPlacement.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEDDRPlan.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEFloatSchedule.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEHotAccumulator.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEPlanIR.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEPointwise.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMERegisterSchedule.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEStreamPlan.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMETargetInfo.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEZeroInit.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "Dialect/BOSCAME/Endpoint.h"
#include "Dialect/BOSCAME/MatrixTransfer.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include <optional>
#include <tuple>

using namespace mlir;
using namespace ::buddy::boscame;

namespace {
DictionaryAttr dict(Operation *op, StringRef name) {
  return op->getAttrOfType<DictionaryAttr>(name);
}
DictionaryAttr dict(DictionaryAttr attr, StringRef name) {
  return attr ? attr.getAs<DictionaryAttr>(name) : DictionaryAttr();
}
StringRef string(DictionaryAttr attr, StringRef name) {
  auto value = attr ? attr.getAs<StringAttr>(name) : StringAttr();
  return value ? value.getValue() : StringRef();
}
int64_t integer(DictionaryAttr attr, StringRef name) {
  auto value = attr ? attr.getAs<IntegerAttr>(name) : IntegerAttr();
  return value ? value.getInt() : -1;
}
int64_t bytes(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth() % 8 ? 0 : integer.getWidth() / 8;
  if (auto floating = dyn_cast<FloatType>(type))
    return floating.getWidth() % 8 ? 0 : floating.getWidth() / 8;
  return 0;
}
Value root(Value value) {
  auto origin = spmdma::getStaticPointerOrigin(value);
  return origin ? origin->root : Value();
}
bool privateRoot(Value value) {
  return value && value.getDefiningOp() &&
         isa<memref::AllocOp, memref::AllocaOp>(value.getDefiningOp());
}
bool argumentRoot(Value value) {
  auto argument = dyn_cast_or_null<BlockArgument>(value);
  return argument && isa<func::FuncOp>(argument.getOwner()->getParentOp());
}
bool nestedPlan(Attribute attribute) {
  if (auto dictionary = dyn_cast<DictionaryAttr>(attribute))
    return llvm::any_of(dictionary, [](NamedAttribute field) {
      return field.getName().strref().starts_with("plan.") ||
             nestedPlan(field.getValue());
    });
  if (auto array = dyn_cast<ArrayAttr>(attribute))
    return llvm::any_of(array, nestedPlan);
  return false;
}
// Offsets are relative to the aligned pointer, in bytes. A view may also
// change that pointer; do not confuse its byte shift with an element offset.
std::optional<std::pair<int64_t, int64_t>> byteBounds(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  auto origin = spmdma::getStaticPointerOrigin(value);
  auto offset = spmdma::getStaticDescriptorOffset(value);
  SmallVector<int64_t> strides;
  int64_t ignored, begin, end;
  int64_t width = type ? bytes(type.getElementType()) : 0;
  if (!type || !type.hasStaticShape() || !width || !origin || !offset ||
      failed(type.getStridesAndOffset(strides, ignored)) ||
      llvm::MulOverflow(*offset, width, begin) ||
      llvm::AddOverflow(begin, origin->byteShift, begin))
    return std::nullopt;
  end = begin;
  for (auto [dim, stride] : llvm::zip(type.getShape(), strides)) {
    int64_t delta;
    if (dim <= 0 || stride <= 0 || llvm::MulOverflow(dim - 1, stride, delta) ||
        llvm::MulOverflow(delta, width, delta) ||
        llvm::AddOverflow(end, delta, end))
      return std::nullopt;
  }
  if (llvm::AddOverflow(end, width, end))
    return std::nullopt;
  return std::make_pair(begin, end);
}
// Placement replaces fresh logical storage with fixed physical addresses.
// Every alias must remain in the owning scope and complete its memory access
// here. Otherwise a retained address or asynchronous operation can outlive the
// scope or observe a later reuse that was impossible for the original root.
bool localCompletedSPMUses(Value storage, Region *scope) {
  SmallVector<Value> work{storage};
  DenseSet<Value> visited;
  while (!work.empty()) {
    Value value = work.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (Operation *user : value.getUsers()) {
      if (!scope->isAncestor(user->getParentRegion()))
        return false;
      if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
        if (view.getViewSource() != value || user->getNumRegions() != 0 ||
            !isMemoryEffectFree(user))
          return false;
        for (Value alias : user->getResults()) {
          if (!isa<BaseMemRefType>(alias.getType()))
            continue;
          auto type = dyn_cast<MemRefType>(alias.getType());
          SmallVector<int64_t> strides;
          int64_t offset;
          if (!type || !type.hasStaticShape() ||
              failed(type.getStridesAndOffset(strides, offset)) ||
              llvm::any_of(strides,
                           [](int64_t stride) { return stride <= 0; }) ||
              !spmdma::getStaticDescriptorOffset(alias) ||
              !spmdma::getStaticPointerOrigin(alias))
            return false;
          work.push_back(alias);
        }
        continue;
      }
      if (isa<memref::CopyOp, memref::LoadOp, memref::StoreOp,
              memref::DeallocOp, memref::DimOp, vector::LoadOp,
              vector::StoreOp>(user))
        continue;
      if (isa<linalg::FillOp, linalg::GenericOp, linalg::MatmulOp,
              linalg::ReduceOp>(user)) {
        bool pure = true;
        for (Region &region : user->getRegions())
          region.walk([&](Operation *payload) {
            if (!isMemoryEffectFree(payload))
              pure = false;
          });
        if (pure)
          continue;
      }
      // Calls, yielded pointers, address extraction and asynchronous effects
      // need an independent escape/completion proof absent from this recipe.
      return false;
    }
  }
  return true;
}
Operation *unsupportedSPMAncestor(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<func::FuncOp>(parent))
      return nullptr;
    if (!isa<scf::ForOp, scf::IfOp>(parent))
      return parent;
  }
  return op;
}
bool exclusive(Operation *a, Operation *b) {
  for (Operation *parent = a->getParentOp(); parent;
       parent = parent->getParentOp())
    if (auto branch = dyn_cast<scf::IfOp>(parent)) {
      Region *aRegion = nullptr, *bRegion = nullptr;
      for (Region &region : branch->getRegions()) {
        if (region.isAncestor(a->getParentRegion()))
          aRegion = &region;
        if (region.isAncestor(b->getParentRegion()))
          bRegion = &region;
      }
      if (aRegion && bRegion && aRegion != bRegion && a->getNumResults() == 1 &&
          b->getNumResults() == 1 &&
          localCompletedSPMUses(a->getResult(0), aRegion) &&
          localCompletedSPMUses(b->getResult(0), bRegion))
        return true;
    }
  return false;
}

class PlanVerifier {
public:
  explicit PlanVerifier(ModuleOp module) : module(module) {}
  LogicalResult run() {
    if (module->hasAttr("bosc_ame.store_policy"))
      return module.emitError(
          "bosc_ame.store_policy is obsolete; remove it and "
          "rerun the automatic BOSCAME planner");
    module.walk([&](Operation *op) { verifySchema(op); });
    if (invalid)
      return failure();
    if (failed(::buddy::contract::verifyKernelContract(module, true)))
      return failure();
    auto version = module->getAttrOfType<IntegerAttr>("plan.version");
    auto profile = module->getAttrOfType<StringAttr>("plan.target_profile");
    require(module,
            version && (version.getInt() == 1 || version.getInt() == 2) &&
                profile && profile.getValue() == "nr-fpga",
            "requires plan.version=1 or 2 and plan.target_profile=nr-fpga");
    if (isV2() &&
        (failed(verifyRegisterSchedules(module)) ||
         failed(verifyStreams(module)) || failed(verifyFloatSchedules(module))))
      return failure();
    for (auto function : module.getOps<func::FuncOp>())
      if (!function.isExternal())
        verifyFunction(function);
    return failure(invalid);
  }

private:
  bool isV2() const {
    auto version = module->getAttrOfType<IntegerAttr>("plan.version");
    return version && version.getInt() == 2;
  }
  void require(Operation *op, bool condition, const Twine &message) {
    if (!condition) {
      op->emitError() << "invalid BOSCAME plan: " << message;
      invalid = true;
    }
  }
  void fields(Operation *op, DictionaryAttr value, ArrayRef<StringRef> integers,
              ArrayRef<StringRef> strings = {},
              ArrayRef<StringRef> booleans = {}) {
    if (!value) {
      require(op, false, "execution plan must be a dictionary");
      return;
    }
    for (NamedAttribute field : value) {
      StringRef key = field.getName().strref();
      if (llvm::is_contained(integers, key)) {
        auto number = dyn_cast<IntegerAttr>(field.getValue());
        require(op, number && number.getType().isInteger(64),
                "plan field must have i64 type: " + key);
      } else if (llvm::is_contained(strings, key)) {
        require(op, isa<StringAttr>(field.getValue()),
                "plan field must have string type: " + key);
      } else if (llvm::is_contained(booleans, key)) {
        require(op, isa<BoolAttr>(field.getValue()),
                "plan field must have bool type: " + key);
      }
    }
  }
  void selectionSchema(Operation *op, DictionaryAttr p) {
    fields(op, p, {"operand", "bank", "offset", "bytes"}, {"level", "reason"});
    require(op,
            p && p.getAs<IntegerAttr>("operand") &&
                p.getAs<StringAttr>("level"),
            "operand selection requires operand and level");
  }
  void verifySchema(Operation *op) {
    if (isa<ModuleOp>(op))
      require(op, op == module.getOperation(),
              "nested modules require separate BOSCAME plans");
    auto logicalMemory = [&](Type type) {
      if (auto memref = dyn_cast<BaseMemRefType>(type)) {
        Attribute space = memref.getMemorySpace();
        auto zero = dyn_cast_or_null<IntegerAttr>(space);
        require(
            op, !space || (zero && zero.getInt() == 0),
            "planned memrefs must retain logical memory space until emission");
      }
    };
    for (Type type : op->getOperandTypes())
      logicalMemory(type);
    for (Type type : op->getResultTypes())
      logicalMemory(type);
    if (auto function = dyn_cast<func::FuncOp>(op))
      for (Type type : function.getArgumentTypes())
        logicalMemory(type);
    for (NamedAttribute attr : op->getAttrs()) {
      require(
          op, !nestedPlan(attr.getValue()),
          "plan attributes must be attached directly to supported operations");
      StringRef name = attr.getName().strref();
      if (!name.starts_with("plan."))
        continue;
      bool carrier =
          ((name == "plan.version" || name == "plan.target_profile") &&
           isa<ModuleOp>(op)) ||
          (name == "plan.automatic_spm" && isV2() && isa<ModuleOp>(op)) ||
          ((name == "plan.budget" || name == "plan.task_map" ||
            name == "plan.source_task_count") &&
           isa<func::FuncOp>(op)) ||
          (isV2() &&
           (name == "plan.stream" || name == "plan.stream_virtual" ||
            name == "plan.register_schedule" ||
            name == "plan.register_member" || name == "plan.register_virtual" ||
            name == "plan.float_schedule" || name == "plan.float_member" ||
            name == "plan.float_virtual")) ||
          (name == "plan.cpu" && isV2()) ||
          (name == "plan.buffer" &&
           isa<memref::AllocOp, memref::AllocaOp>(op)) ||
          (name == "plan.fixed_input" && isV2() &&
           isa<memref::AllocOp, memref::AllocaOp>(op)) ||
          (name == "plan.transfer" && isa<memref::CopyOp>(op)) ||
          (name == "plan.route" && isa<linalg::MatmulOp>(op)) ||
          (name == "plan.compute" && isa<linalg::MatmulOp>(op)) ||
          (name == "plan.fill" && isa<linalg::FillOp>(op)) ||
          (name == "plan.vector" && isa<linalg::GenericOp>(op)) ||
          (name == "plan.reduce" && isa<linalg::ReduceOp>(op)) ||
          (name == "plan.scalar" && isa<memref::LoadOp>(op)) ||
          (name == "plan.loop" && isa<scf::ForOp>(op)) ||
          (name == "plan.scope" && isa<scf::IfOp>(op));
      require(op, carrier, "unknown plan or unsupported carrier: " + name);
      if (isV2() &&
          (name == "plan.stream" || name == "plan.stream_virtual" ||
           name == "plan.register_schedule" || name == "plan.register_member" ||
           name == "plan.register_virtual" || name == "plan.float_schedule" ||
           name == "plan.float_member" || name == "plan.float_virtual"))
        continue; // Dedicated verifier reconstructs the complete recipe.
      if (name == "plan.version") {
        auto version = dyn_cast<IntegerAttr>(attr.getValue());
        require(op, version && version.getType().isInteger(32),
                "plan.version must have i32 type");
        continue;
      }
      if (name == "plan.target_profile") {
        require(op, isa<StringAttr>(attr.getValue()),
                "plan.target_profile must have string type");
        continue;
      }
      if (name == "plan.automatic_spm") {
        require(op, isa<BoolAttr>(attr.getValue()),
                "plan.automatic_spm must have bool type");
        continue;
      }
      if (name == "plan.source_task_count") {
        auto count = dyn_cast<IntegerAttr>(attr.getValue());
        require(op,
                count && count.getType().isInteger(64) && count.getInt() >= 0,
                "plan.source_task_count must be a nonnegative i64");
        continue;
      }
      if (name == "plan.task_map") {
        auto mapping = dyn_cast<ArrayAttr>(attr.getValue());
        require(op, bool(mapping), "plan.task_map must be an array");
        if (mapping)
          for (Attribute entry : mapping)
            fields(op, dyn_cast<DictionaryAttr>(entry), {"old_id", "new_id"},
                   {"action"});
        continue;
      }
      if (name == "plan.fixed_input") {
        auto fixed = dyn_cast<DictionaryAttr>(attr.getValue());
        fields(op, fixed, {"bank", "offset", "capacity"});
        require(op,
                fixed && fixed.size() == 3 &&
                    fixed.getAs<IntegerAttr>("bank") &&
                    fixed.getAs<IntegerAttr>("offset") &&
                    fixed.getAs<IntegerAttr>("capacity"),
                "fixed input requires exactly bank, offset and capacity");
        if (fixed) {
          int64_t bank = integer(fixed, "bank");
          int64_t offset = integer(fixed, "offset");
          int64_t capacity = integer(fixed, "capacity");
          require(op,
                  bank >= 0 && bank < 2 && offset >= 0 && offset % 64 == 0 &&
                      capacity > 0 && capacity <= 131072 &&
                      offset <= 131072 - capacity,
                  "fixed input exceeds the aligned nr-fpga SPM window");
        }
        continue;
      }
      auto p = dyn_cast<DictionaryAttr>(attr.getValue());
      fields(op, p,
             {"bank",
              "offset",
              "capacity",
              "channel",
              "vector_bytes",
              "rows",
              "row_bytes",
              "elements",
              "axis",
              "trips",
              "spm_banks",
              "bank_capacity",
              "completion",
              "total_bytes",
              "temp_bytes",
              "consumer",
              "lanes",
              "alignment",
              "loop_bytes",
              "boundary_bytes",
              "pitch",
              "ddr_cycles_x2",
              "spm_cycles_x2",
              "margin_cycles_x2",
              "saving_cycles_x2"},
             {"level",    "role",
              "engine",   "from",
              "to",       "guard",
              "wait",     "kind",
              "tail",     "fp_policy",
              "strategy", "reason",
              "order",    "matrix_rejection",
              "recipe",   "temporary_window",
              "ordering", "profile",
              "evidence", "mode"},
             {"spm_window_reuse", "share_spm_windows"});
      if (!p)
        continue;
      if (name == "plan.buffer")
        require(op, p.getAs<StringAttr>("role") && p.getAs<StringAttr>("level"),
                "buffer placement requires level and role");
      if (name == "plan.budget")
        require(op, bool(p.getAs<DenseI64ArrayAttr>("high_water")),
                "plan budget high_water must be an i64 array");
      if (name == "plan.vector" && p.get("scalar_operands")) {
        auto scalars = p.getAs<ArrayAttr>("scalar_operands");
        require(op, bool(scalars),
                "pointwise scalar_operands must be an array");
        if (scalars)
          for (Attribute entry : scalars) {
            auto scalar = dyn_cast<DictionaryAttr>(entry);
            fields(op, scalar, {"operation", "operand"}, {"kind"});
            require(op,
                    scalar && scalar.getAs<IntegerAttr>("operand") &&
                        scalar.getAs<StringAttr>("kind") &&
                        isa_and_nonnull<IntegerAttr, FloatAttr>(
                            scalar.get("value")),
                    "pointwise scalar broadcast requires operand, kind and "
                    "a scalar typed constant value");
          }
      }
      if (name == "plan.vector" && p.get("math")) {
        auto recipes = p.getAs<ArrayAttr>("math");
        require(op, bool(recipes), "pointwise math must be a recipe array");
        if (recipes)
          for (Attribute entry : recipes)
            fields(op, dyn_cast<DictionaryAttr>(entry), {"operation"},
                   {"function", "recipe", "evaluation"});
      }
      if (name == "plan.transfer")
        fields(op, dict(p, "chunks"),
               {"count", "bytes", "ddr_stride", "spm_stride", "source_stride",
                "target_stride"});
      if (name == "plan.transfer")
        require(op, bool(p.getAs<StringAttr>("role")),
                "transfer plan requires a role");
      if (name == "plan.transfer" && string(p, "engine") == "matrix") {
        if (string(p, "role") == "internal")
          for (StringRef key : {"source_endpoint", "target_endpoint"}) {
            auto endpoint = dict(p, key);
            fields(op, endpoint,
                   {"operand", "bank", "window_offset", "capacity", "begin",
                    "end"},
                   {"level"});
            bool complete = bool(endpoint);
            if (endpoint) {
              for (StringRef field : {"operand", "bank", "window_offset",
                                      "capacity", "begin", "end"})
                complete &= bool(endpoint.getAs<IntegerAttr>(field));
              complete &= bool(endpoint.getAs<StringAttr>("level"));
            }
            require(op, complete,
                    "internal copy requires complete physical endpoints");
          }
        auto tiles = p.getAs<ArrayAttr>("tiles");
        require(
            op,
            tiles && !tiles.empty() && p.getAs<IntegerAttr>("completion") &&
                p.getAs<IntegerAttr>("total_bytes") &&
                p.getAs<IntegerAttr>("temp_bytes"),
            "matrix transfer requires tiles, completion and byte accounting");
        for (StringRef key : {"source_layout", "target_layout"}) {
          auto layout = dict(p, key);
          fields(op, layout,
                 {"element_bits", "descriptor_offset", "contiguous_axis",
                  "rows", "row_bytes", "row_pitch"},
                 {"origin"});
          require(op,
                  layout && layout.getAs<DenseI64ArrayAttr>("shape") &&
                      layout.getAs<DenseI64ArrayAttr>("strides"),
                  "matrix layout requires logical shape and strides");
          if (layout)
            for (StringRef field :
                 {"element_bits", "descriptor_offset", "contiguous_axis",
                  "rows", "row_bytes", "row_pitch"})
              require(op, bool(layout.getAs<IntegerAttr>(field)),
                      "matrix layout is missing a required field");
        }
        if (tiles)
          for (Attribute attribute : tiles) {
            auto tile = dyn_cast<DictionaryAttr>(attribute);
            fields(op, tile,
                   {"id", "predecessor", "source_offset", "target_offset",
                    "rows", "row_bytes", "source_pitch", "target_pitch",
                    "instruction_bits", "register_count", "mtype"},
                   {"register_class"});
            bool complete = bool(tile);
            if (tile) {
              for (StringRef field :
                   {"id", "predecessor", "source_offset", "target_offset",
                    "rows", "row_bytes", "source_pitch", "target_pitch",
                    "instruction_bits", "register_count", "mtype"})
                complete &= bool(tile.getAs<IntegerAttr>(field));
              complete &= bool(tile.getAs<StringAttr>("register_class"));
            }
            require(op, complete, "matrix tile is missing a required field");
          }
      }
      if (name == "plan.transfer" && string(p, "engine") == "segmented") {
        auto segments = p.getAs<ArrayAttr>("segments");
        require(op,
                segments && !segments.empty() &&
                    p.getAs<IntegerAttr>("completion"),
                "segmented transfer requires segments and completion");
        if (segments)
          for (Attribute attribute : segments) {
            auto segment = dyn_cast<DictionaryAttr>(attribute);
            fields(op, segment,
                   {"id", "predecessor", "source_offset", "target_offset",
                    "bytes", "vector_bytes"},
                   {"engine", "reason"});
            require(op,
                    segment && segment.getAs<IntegerAttr>("id") &&
                        segment.getAs<IntegerAttr>("predecessor") &&
                        segment.getAs<IntegerAttr>("source_offset") &&
                        segment.getAs<IntegerAttr>("target_offset") &&
                        segment.getAs<IntegerAttr>("bytes") &&
                        segment.getAs<StringAttr>("engine") &&
                        segment.getAs<StringAttr>("reason"),
                    "transfer segment is missing a required field");
          }
      }
      if (name == "plan.route") {
        fields(op, dict(p, "tile"), {"m", "n", "k"});
        fields(op, dict(p, "subtiles"), {"n", "k"});
        for (StringRef key : {"a", "b", "accumulator", "output"})
          selectionSchema(op, dict(p, key));
        if (p.get("initialization")) {
          auto init = dict(p, "initialization");
          fields(op, init, {"task"}, {"kind"});
          require(op,
                  init && init.getAs<IntegerAttr>("task") &&
                      init.getAs<StringAttr>("kind"),
                  "zero initialization requires kind and producer task");
        }
      }
      if (name == "plan.compute")
        require(op, bool(p.getAs<DenseI64ArrayAttr>("shape")),
                "scalar matmul shape must be an i64 array");
      if (name == "plan.vector" || name == "plan.compute") {
        auto operands = p.getAs<ArrayAttr>("operands");
        require(op, bool(operands), "compute operands must be an array");
        if (operands)
          for (Attribute operand : operands)
            selectionSchema(op, dyn_cast<DictionaryAttr>(operand));
      }
      if (name == "plan.reduce") {
        selectionSchema(op, dict(p, "input"));
        selectionSchema(op, dict(p, "output"));
      }
      if (name == "plan.scalar")
        selectionSchema(op, dict(p, "input"));
    }
  }
  DictionaryAttr placement(Value value) {
    auto found = placements.find(root(value));
    return found == placements.end() ? DictionaryAttr() : found->second;
  }
  bool staticType(Value value, SmallVectorImpl<int64_t> &strides) {
    auto type = dyn_cast<MemRefType>(value.getType());
    int64_t offset;
    return type && type.hasStaticShape() &&
           succeeded(type.getStridesAndOffset(strides, offset)) &&
           llvm::all_of(strides, [](int64_t s) { return s > 0; }) &&
           bytes(type.getElementType()) > 0;
  }
  bool alignedRoot(Value value, int64_t alignmentBytes = 64) {
    if (alignmentBytes <= 0)
      return false;
    Value storage = root(value);
    if (privateRoot(storage)) {
      auto p = placement(value);
      return (string(p, "level") == "spm" &&
              integer(p, "offset") % alignmentBytes == 0) ||
             (isV2() && string(p, "level") == "ddr" &&
              integer(p, "alignment") >= alignmentBytes &&
              integer(p, "alignment") % alignmentBytes == 0);
    }
    auto abi = dict(module, "contract.abi");
    int64_t alignment = integer(abi, "args_aligned_bytes");
    return argumentRoot(storage) && alignment >= alignmentBytes &&
           alignment % alignmentBytes == 0 &&
           buddy::contract::hasEntryArgumentABI(
               storage.getParentRegion()->getParentOfType<func::FuncOp>());
  }
  bool aligned(Value value, int64_t alignmentBytes = 64) {
    return isa<MemRefType>(value.getType()) &&
           spmdma::isEndpointPointerAlignmentProven(value, alignmentBytes) &&
           alignedRoot(value, alignmentBytes);
  }
  void bounds(Operation *op, Value value) {
    auto p = placement(value);
    auto range = byteBounds(value);
    if (string(p, "level") == "spm" ||
        (isV2() && string(p, "level") == "ddr")) {
      require(
          op,
          range && range->first >= 0 && range->second <= integer(p, "capacity"),
          "SPM view exceeds its planned capacity or is not statically bounded");
    } else if (argumentRoot(root(value))) {
      auto rootType = dyn_cast<MemRefType>(root(value).getType());
      // Unranked ABI arguments do not carry allocation extents. The frontend
      // provides a valid effective view; DMA coverage is checked against it.
      if (rootType && rootType.hasStaticShape()) {
        auto storage = byteBounds(root(value));
        require(op,
                range && storage && range->first >= storage->first &&
                    range->second <= storage->second,
                "DDR view exceeds its statically known argument storage");
      }
    }
  }
  void binding(Operation *op, DictionaryAttr selection, unsigned index) {
    if (!selection || index >= op->getNumOperands()) {
      require(op, false, "missing route operand binding");
      return;
    }
    require(op, integer(selection, "operand") == index,
            "route operand index does not match the actual operation");
    Value value = op->getOperand(index);
    auto p = placement(value);
    StringRef level = string(selection, "level");
    if (level == "scalar") {
      require(op,
              supportedPointwiseScalar(value.getType()) &&
                  string(selection, "reason") == "explicit-scalar-input",
              "scalar route requires an explicit scalar operand");
    } else if (level == "ddr") {
      auto kind = classifyEndpoint(value);
      require(op,
              kind == EndpointKind::ExternalDDR ||
                  (isV2() && string(p, "level") == "ddr" &&
                   kind == EndpointKind::PrivateAllocation),
              "DDR operand requires an external entry argument or a verified "
              "private frame candidate");
      if (kind == EndpointKind::PrivateAllocation)
        require(op, succeeded(verifyPrivateEndpoint(op, value)),
                "private AME DDR endpoint lacks a bounded invocation-local "
                "frame proof");
      require(op, aligned(value),
              "direct DDR address is not proven 64-byte aligned");
      bounds(op, value);
    } else if (level == "host") {
      auto type = dyn_cast<MemRefType>(value.getType());
      require(op, string(p, "level") == "host" && type && type.getRank() == 0,
              "host route requires a planned rank-zero buffer");
    } else if (level == "spm") {
      require(op,
              string(p, "level") == "spm" &&
                  integer(selection, "bank") == integer(p, "bank") &&
                  integer(selection, "offset") == integer(p, "offset"),
              "SPM route does not match its root placement");
      require(op, aligned(value), "SPM operand is not proven 64-byte aligned");
      bounds(op, value);
    } else {
      require(op, false, "unsupported operand route");
    }
  }
  void
  verifyMeasuredAccumulatorRecipe(Value storage,
                                  const MeasuredAccumulatorProfile &profile) {
    auto function = storage.getDefiningOp()->getParentOfType<func::FuncOp>();
    Builder builder(module.getContext());
    auto i = [&](StringRef key, int64_t value) {
      return builder.getNamedAttr(key, builder.getI64IntegerAttr(value));
    };
    auto s = [&](StringRef key, StringRef value) {
      return builder.getNamedAttr(key, builder.getStringAttr(value));
    };
    auto d = [&](ArrayRef<NamedAttribute> fields) {
      return builder.getDictionaryAttr(fields);
    };
    auto binding = [&](int64_t operand, int64_t size, bool spm,
                       StringRef reason) {
      NamedAttrList fields({i("operand", operand), i("bytes", size),
                            s("level", spm ? "spm" : "ddr"),
                            s("reason", reason)});
      if (spm) {
        fields.append(i("bank", 0));
        fields.append(i("offset", 0));
      }
      return fields.getDictionary(module.getContext());
    };
    DenseMap<Operation *, std::pair<StringRef, DictionaryAttr>> recipes;
    auto expect = [&](Operation *op, StringRef name, DictionaryAttr expected) {
      recipes[op] = {name, expected};
      require(op, dict(op, name) == expected,
              "measured hot accumulator requires the exact calibrated "
              "execution recipe: " +
                  name);
    };
    linalg::MatmulOp matmul;
    linalg::FillOp fill;
    for (Operation *user : storage.getUsers()) {
      if (auto candidate = dyn_cast<linalg::MatmulOp>(user))
        matmul = candidate;
      if (auto candidate = dyn_cast<linalg::FillOp>(user))
        fill = candidate;
    }
    require(storage.getDefiningOp(), bool(matmul) && bool(fill),
            "measured hot accumulator lost its source producer or fill");
    if (!matmul || !fill)
      return;
    bool normal = profile.mode == MeasuredAccumulatorMode::NormalPostAdd;
    NamedAttrList route(
        {s("engine", "ame"),
         builder.getNamedAttr("tile", d({i("m", 32), i("n", 64), i("k", 64)})),
         builder.getNamedAttr("subtiles", d({i("n", 1), i("k", 2)})),
         builder.getNamedAttr(
             "a", binding(0, 4096, false, "legal-minimum-external-traffic")),
         builder.getNamedAttr(
             "b", binding(1, 8192, false, "legal-minimum-external-traffic")),
         builder.getNamedAttr("accumulator",
                              binding(2, 8192, true, "private-accumulator")),
         builder.getNamedAttr(
             "output", binding(2, 8192, true, "same-accumulator-window"))});
    if (normal) {
      expect(fill, "plan.fill",
             d({s("level", "spm"), s("engine", "rvv"), i("vector_bytes", 64),
                i("rows", 32), i("row_bytes", 256), s("kind", "overwrite")}));
    } else {
      int64_t fillID = integer(dict(fill, "contract.task"), "id");
      int64_t matmulID = integer(dict(matmul, "contract.task"), "id");
      require(fill, fillID >= 0 && matmulID >= 0,
              "measured deferred zero requires its source task identities");
      expect(fill, "plan.fill",
             d({s("level", "spm"), s("engine", "ame"),
                s("kind", "deferred-zero"), i("consumer", matmulID)}));
      route.append(builder.getNamedAttr(
          "initialization",
          d({s("kind", "first-iteration-zero"), i("task", fillID)})));
    }
    expect(matmul, "plan.route", route.getDictionary(module.getContext()));
    auto loop = cast<scf::ForOp>(matmul->getParentOp());
    expect(loop, "plan.loop",
           d({i("trips", profile.trips),
              builder.getNamedAttr("spm_window_reuse",
                                   builder.getBoolAttr(true))}));
    for (Operation &op : function.getBody().front()) {
      if (auto copy = dyn_cast<memref::CopyOp>(op)) {
        if (copy.getSource() != storage) {
          auto recipe = getDDRRecipe(copy);
          require(
              copy, normal && succeeded(recipe),
              "measured old-C snapshot requires the scalar DDR copy recipe");
          if (succeeded(recipe))
            expect(copy, "plan.cpu", *recipe);
          continue;
        }
        auto destination =
            cast<memref::ReinterpretCastOp>(copy.getTarget().getDefiningOp());
        int64_t ddrStride = destination.getSource() == function.getArgument(2)
                                ? profile.pitch * 4
                                : 256;
        expect(copy, "plan.transfer",
               d({s("role", "output"), s("engine", "dma"), i("channel", 0),
                  s("from", "spm"), s("to", "ddr"),
                  builder.getNamedAttr(
                      "chunks",
                      d({i("count", 32), i("bytes", 256),
                         i("ddr_stride", ddrStride), i("spm_stride", 256)})),
                  s("wait", "after-each-chunk"),
                  s("reason", "bounded-external-boundary")}));
      } else if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
        SmallVector<Attribute> operands;
        operands.push_back(binding(0, 8192, true, "pointwise-memory-operand"));
        operands.push_back(binding(1, 8192, false, "pointwise-memory-operand"));
        operands.push_back(binding(2, 8192, true, "pointwise-memory-operand"));
        require(generic, normal,
                "overwrite measured recipe cannot contain a post-add");
        expect(
            generic, "plan.vector",
            d({s("engine", "rvv"), i("vector_bytes", 64), i("elements", 2048),
               i("lanes", 16), s("tail", "none"),
               builder.getNamedAttr("operands", builder.getArrayAttr(operands)),
               s("fp_policy", "preserve")}));
      }
    }
    // Equivalent but differently tiled or competing recipes are outside the
    // measured complete-adapter table, even when the generic verifier could
    // establish their standalone legality.
    function.walk([&](Operation *op) {
      if (op == function)
        return;
      auto expected = recipes.find(op);
      for (NamedAttribute attr : op->getAttrs()) {
        StringRef name = attr.getName().strref();
        if (!name.starts_with("plan."))
          continue;
        bool allowed =
            isa<memref::AllocOp, memref::AllocaOp>(op)
                ? name == "plan.buffer"
                : expected != recipes.end() && name == expected->second.first;
        require(op, allowed,
                "measured hot accumulator cannot contain an additional "
                "or alternative execution recipe: " +
                    name);
      }
    });
  }
  void verifyFunction(func::FuncOp function) {
    placements.clear();
    SmallVector<Operation *> allocations;
    DenseMap<Operation *, DictionaryAttr> fixedPlacements;
    int64_t reservedHighWater[2] = {0, 0};
    bool automaticSPM = true;
    if (auto automatic = module->getAttrOfType<BoolAttr>("plan.automatic_spm"))
      automaticSPM = automatic.getValue();
    if (isV2()) {
      Builder builder(module.getContext());
      // Target constraints are reserved before any automatic candidate,
      // regardless of their source order or the lifetime of a hot P tile.
      function.walk([&](Operation *op) {
        if (!isa<memref::AllocOp, memref::AllocaOp>(op))
          return;
        auto fixed = dict(op, "plan.fixed_input");
        auto p = dict(op, "plan.buffer");
        if (!fixed) {
          require(op, string(p, "role") != "fixed-input",
                  "fixed-input placement requires its target input record");
          return;
        }
        int64_t bank = integer(fixed, "bank");
        int64_t offset = integer(fixed, "offset");
        int64_t capacity = integer(fixed, "capacity");
        auto expected = builder.getDictionaryAttr(
            {builder.getNamedAttr("level", builder.getStringAttr("spm")),
             builder.getNamedAttr("bank", builder.getI64IntegerAttr(bank)),
             builder.getNamedAttr("offset", builder.getI64IntegerAttr(offset)),
             builder.getNamedAttr("capacity",
                                  builder.getI64IntegerAttr(capacity)),
             builder.getNamedAttr("role",
                                  builder.getStringAttr("fixed-input"))});
        fixedPlacements[op] = expected;
        require(
            op, p == expected,
            "fixed target input requires its exact canonical SPM placement");
        int64_t end;
        bool valid = bank >= 0 && bank < 2 && offset >= 0 && offset % 64 == 0 &&
                     capacity > 0 && capacity <= 131072 &&
                     !llvm::AddOverflow(offset, capacity, end) && end <= 131072;
        require(op, valid, "fixed target input has invalid SPM bounds");
        if (valid)
          reservedHighWater[bank] = std::max(reservedHighWater[bank], end);
      });
    }
    function.walk([&](Operation *op) {
      if (!isa<memref::AllocOp, memref::AllocaOp>(op))
        return;
      auto p = dict(op, "plan.buffer");
      auto type = dyn_cast<MemRefType>(op->getResult(0).getType());
      require(op, p && type && type.hasStaticShape(),
              "every private allocation needs a static plan.buffer");
      if (!p || !type || !type.hasStaticShape())
        return;
      placements[op->getResult(0)] = p;
      allocations.push_back(op);
      int64_t capacity = integer(p, "capacity");
      SmallVector<int64_t> strides;
      int64_t offset, needed = 1;
      bool legal = bytes(type.getElementType()) &&
                   succeeded(type.getStridesAndOffset(strides, offset)) &&
                   offset == 0;
      for (auto [dim, stride] : llvm::zip(type.getShape(), strides)) {
        int64_t delta;
        if (dim <= 0 || stride <= 0 ||
            llvm::MulOverflow(dim - 1, stride, delta) ||
            llvm::AddOverflow(needed, delta, needed))
          legal = false;
      }
      legal &= !llvm::MulOverflow(needed, bytes(type.getElementType()), needed);
      require(op, legal && capacity >= needed,
              "planned buffer capacity is smaller than its storage layout");
      StringRef level = string(p, "level");
      bool virtualRoot =
          isV2() && (isStreamVirtual(op) || isRegisterScheduleVirtual(op) ||
                     isFloatScheduleVirtual(op));
      if (isV2()) {
        bool hotClaim = string(p, "role") == "hot-accumulator";
        for (StringRef key :
             {"loop_bytes", "boundary_bytes", "profile", "evidence", "mode",
              "pitch", "ddr_cycles_x2", "spm_cycles_x2", "margin_cycles_x2",
              "saving_cycles_x2"})
          hotClaim |= bool(p.get(key));
        if (!fixedPlacements.count(op) && !virtualRoot) {
          // A fixed old-C snapshot on bank1 changes the measured epilogue
          // even when bank0/offset0 is free. The first table represents no
          // additional fixed workspace in either bank.
          auto expected = getHotAccumulatorPlacement(
              op->getResult(0), automaticSPM,
              reservedHighWater[0] == 0 && fixedPlacements.empty());
          if (expected) {
            require(op, p == expected,
                    "hot accumulator placement must equal its canonical "
                    "source proof, measured profile and supported window");
            if (string(expected, "level") == "spm") {
              auto profile = analyzeMeasuredAccumulator(op->getResult(0));
              require(op, bool(profile),
                      "positive measured placement requires its exact source "
                      "profile");
              if (profile)
                verifyMeasuredAccumulatorRecipe(op->getResult(0), *profile);
            }
          } else {
            require(op, !hotClaim,
                    "hot accumulator claim has no closed source reuse proof");
            // Unselected workspaces are part of the measured surrounding
            // recipe too. In particular normal old-C snapshots must not be
            // relabelled into unmeasured SPM work by editing their plan.
            auto size = getStaticStorageBytes(op->getResult(0));
            require(op, bool(size),
                    "ordinary V2 workspace requires bounded static storage");
            if (size) {
              Builder builder(module.getContext());
              auto workspace = builder.getDictionaryAttr(
                  {builder.getNamedAttr("level", builder.getStringAttr("ddr")),
                   builder.getNamedAttr("capacity",
                                        builder.getI64IntegerAttr(*size)),
                   builder.getNamedAttr("alignment",
                                        builder.getI64IntegerAttr(64)),
                   builder.getNamedAttr("role",
                                        builder.getStringAttr("workspace")),
                   builder.getNamedAttr(
                       "reason",
                       builder.getStringAttr("reuse-benefit-unproven"))});
              require(op, p == workspace,
                      "ordinary V2 workspace must retain its canonical DDR "
                      "placement until a measured recipe selects it");
            }
          }
        } else {
          require(op, !hotClaim,
                  "fixed or virtual storage cannot claim automatic hot "
                  "accumulator placement");
        }
      }
      if (virtualRoot)
        require(op, level == "virtual",
                "execution recipe cannot consume a materialized or fixed "
                "storage root as a virtual value");
      if (isV2() && level == "virtual") {
        require(op,
                isStreamVirtual(op) || isRegisterScheduleVirtual(op) ||
                    isFloatScheduleVirtual(op),
                "virtual root must belong to a verified execution recipe");
      } else if (isV2() && level == "ddr") {
        require(op,
                legal && integer(p, "alignment") == 64 &&
                    op->getAttrOfType<IntegerAttr>("alignment") &&
                    op->getAttrOfType<IntegerAttr>("alignment").getInt() >= 64,
                "private DDR storage requires a bounded aligned frame slot");
        // Hoisting/reusing a frame slot is safe only for synchronous local
        // accesses. Captured addresses would retain an iteration's identity.
        require(op,
                localCompletedSPMUses(op->getResult(0), &function.getBody()),
                "private DDR frame slot requires non-escaping completed uses");
        // A single entry frame slot cannot replace one allocation per
        // concurrent iteration. Unknown region scopes provide no such proof.
        if (Operation *ancestor = unsupportedSPMAncestor(op))
          require(op, false,
                  "private DDR frame slot requires serial execution inside "
                  "func.func; unsupported ancestor " +
                      ancestor->getName().getStringRef());
      } else if (level == "host") {
        require(op, type.getRank() == 0 && legal,
                "host placement only supports static rank-zero storage");
        for (Operation *parent = op->getParentOp();
             parent && parent != function; parent = parent->getParentOp())
          require(
              op, isa<scf::IfOp>(parent),
              "host allocation in a repeated or unknown scope is unsupported");
        Value storage = op->getResult(0);
        function.walk([&](Operation *user) {
          for (Value operand : user->getOperands()) {
            if (!isa<BaseMemRefType>(operand.getType()) ||
                root(operand) != storage)
              continue;
            bool allowed = isa<memref::LoadOp, memref::DeallocOp>(user);
            if (auto fill = dyn_cast<linalg::FillOp>(user))
              allowed = fill.getOutputs()[0] == operand;
            if (auto reduce = dyn_cast<linalg::ReduceOp>(user))
              allowed = llvm::is_contained(reduce.getInits(), operand);
            require(
                user, allowed,
                "host rank-zero allocation must remain private and not escape");
          }
        });
      } else {
        int64_t bank = integer(p, "bank"), offset = integer(p, "offset");
        require(op,
                level == "spm" && bank >= 0 && bank < 2 && offset >= 0 &&
                    offset % 64 == 0 && capacity > 0 && capacity <= 131072 &&
                    offset <= 131072 - capacity,
                "SPM bank, offset, or capacity exceeds nr-fpga limits");
        require(op,
                localCompletedSPMUses(op->getResult(0), &function.getBody()),
                "SPM root requires non-escaping aliases and completed local "
                "memory accesses");
      }
    });
    if (isV2())
      function.walk([&](Operation *op) {
        bool privateDDR = llvm::is_contained(allocations, op) &&
                          string(dict(op, "plan.buffer"), "level") == "ddr";
        for (Value operand : op->getOperands())
          if (isa<BaseMemRefType>(operand.getType()) &&
              string(placements.lookup(root(operand)), "level") == "ddr")
            privateDDR = true;
        if (privateDDR)
          if (Operation *ancestor = unsupportedSPMAncestor(op))
            require(op, false,
                    "private DDR frame slot requires serial completed "
                    "accesses; unsupported ancestor " +
                        ancestor->getName().getStringRef());
      });
    // Only AME-accessed DDR roots consume the managed operand-window frame.
    // Recompute the same unique aligned slots as emission, independent of
    // metadata. Ordinary CPU-only workspaces retain their existing storage.
    if (isV2()) {
      DenseSet<Value> frameRoots;
      int64_t frameBytes = 0;
      auto account = [&](Operation *op, Value operand) {
        if (classifyEndpoint(operand) != EndpointKind::PrivateAllocation)
          return;
        Value storage = getEndpointRoot(operand);
        if (!frameRoots.insert(storage).second)
          return;
        auto size = getStaticStorageBytes(storage);
        int64_t slot = size && *size <= kPrivateDDRFrameMaxBytes
                           ? llvm::alignTo(*size, int64_t(64))
                           : kPrivateDDRFrameMaxBytes + 1;
        require(op,
                size && slot <= kPrivateDDRFrameMaxBytes &&
                    !llvm::AddOverflow(frameBytes, slot, frameBytes) &&
                    frameBytes <= kPrivateDDRFrameMaxBytes,
                "aggregate private AME DDR frame exceeds guarded "
                "operand-window capacity");
      };
      function.walk([&](Operation *op) {
        if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
          auto route = dict(matmul, "plan.route");
          if (!route)
            return;
          for (auto [name, operand] :
               llvm::zip(ArrayRef<StringRef>{"a", "b", "accumulator"},
                         matmul->getOperands()))
            if (string(dict(route, name), "level") == "ddr")
              account(op, operand);
        } else if (auto copy = dyn_cast<memref::CopyOp>(op)) {
          auto transfer = dict(copy, "plan.transfer");
          if (string(transfer, "from") == "ddr")
            account(op, copy.getSource());
          if (string(transfer, "to") == "ddr")
            account(op, copy.getTarget());
        }
      });
    }
    // Private logical roots are disjoint before placement, but a callee may
    // use the same physical SPM addresses. V1 has no call-frame/workspace
    // contract, including for opaque callees, so do not turn that absence of
    // logical aliasing into an interprocedural physical ownership proof.
    bool hasSPMWorkspace = llvm::any_of(allocations, [&](Operation *op) {
      return string(dict(op, "plan.buffer"), "level") == "spm";
    });
    if (hasSPMWorkspace) {
      require(function, SymbolTable::symbolKnownUseEmpty(function, module),
              "SPM function has module-local symbol uses; v1 has no "
              "interprocedural workspace ownership proof");
      function.walk([&](Operation *op) {
        if (isa<func::CallOp, func::CallIndirectOp>(op))
          require(op, false,
                  "call may clobber live SPM workspace; v1 has no "
                  "interprocedural workspace proof");
        // A synchronous operation can still race another iteration. Fixed
        // addresses are invocation-local only in the supported serial scopes,
        // including when the allocation itself is outside the concurrent op.
        bool usesSPM = llvm::is_contained(allocations, op) &&
                       string(dict(op, "plan.buffer"), "level") == "spm";
        for (Value operand : op->getOperands())
          if (isa<BaseMemRefType>(operand.getType()) &&
              string(placements.lookup(root(operand)), "level") == "spm")
            usesSPM = true;
        if (usesSPM)
          if (Operation *ancestor = unsupportedSPMAncestor(op))
            require(op, false,
                    "fixed SPM workspace requires serial execution inside "
                    "func.func; unsupported ancestor " +
                        ancestor->getName().getStringRef());
      });
    }
    int64_t highWater[2] = {0, 0};
    for (auto [i, op] : llvm::enumerate(allocations)) {
      auto a = dict(op, "plan.buffer");
      if (string(a, "level") != "spm")
        continue;
      int64_t bank = integer(a, "bank"), start = integer(a, "offset");
      int64_t end;
      if (llvm::AddOverflow(start, integer(a, "capacity"), end)) {
        require(op, false, "SPM window end overflows");
        continue;
      }
      if (bank >= 0 && bank < 2)
        highWater[bank] = std::max(highWater[bank], end);
      for (Operation *other : ArrayRef(allocations).drop_front(i + 1)) {
        auto b = dict(other, "plan.buffer");
        if (string(b, "level") != "spm" || bank != integer(b, "bank"))
          continue;
        int64_t bStart = integer(b, "offset"), bEnd;
        if (llvm::AddOverflow(bStart, integer(b, "capacity"), bEnd)) {
          require(other, false, "SPM window end overflows");
          continue;
        }
        require(op, end <= bStart || bEnd <= start || exclusive(op, other),
                "overlapping SPM windows require mutually exclusive scopes");
      }
    }
    auto budget = dict(function, "plan.budget");
    auto water = budget ? budget.getAs<DenseI64ArrayAttr>("high_water")
                        : DenseI64ArrayAttr();
    require(function,
            integer(budget, "spm_banks") == 2 &&
                integer(budget, "bank_capacity") == 131072 && water &&
                water.asArrayRef() == ArrayRef<int64_t>(highWater),
            "plan.budget does not match the physical windows");
    DenseSet<int64_t> taskIDs, oldIDs;
    function.walk([&](Operation *op) {
      if (auto task = dict(op, "contract.task"))
        taskIDs.insert(integer(task, "id"));
    });
    auto mapping = function->getAttrOfType<ArrayAttr>("plan.task_map");
    auto sourceCount =
        function->getAttrOfType<IntegerAttr>("plan.source_task_count");
    if (mapping || sourceCount)
      require(function,
              mapping && sourceCount && sourceCount.getInt() >= 0 &&
                  uint64_t(sourceCount.getInt()) == mapping.size(),
              "source task mapping coverage mismatch");
    if (mapping)
      for (Attribute entry : mapping) {
        auto item = cast<DictionaryAttr>(entry);
        int64_t oldID = integer(item, "old_id");
        StringRef action = string(item, "action");
        require(function, oldID >= 0 && oldIDs.insert(oldID).second,
                "source task mapping IDs must be nonnegative and unique");
        require(function,
                action == "retained" || action == "folded-load" ||
                    action == "folded-store" || action == "eliminated",
                "unsupported source task mapping action");
        require(function,
                action == "eliminated"
                    ? !item.get("new_id")
                    : taskIDs.contains(integer(item, "new_id")),
                "invalid source task mapping target");
      }
    function.walk([&](Operation *op) { verifyOperation(op); });
  }
  void verifyOperation(Operation *op) {
    if (isV2() && (isStreamMember(op) || isRegisterScheduleMember(op) ||
                   isFloatScheduleMember(op)))
      return;
    if (isa<memref::AllocOp, memref::AllocaOp>(op))
      return;
    if (op->hasAttr("plan.cpu")) {
      require(op, isV2() && succeeded(verifyDDRRecipe(op)),
              "invalid v2 DDR computation");
      require(op, !isa<linalg::FillOp>(op) || !op->hasAttr("plan.fill"),
              "a real CPU fill and a deferred/vector fill are competing "
              "execution recipes");
      for (Value operand : op->getOperands())
        if (isa<BaseMemRefType>(operand.getType()))
          bounds(op, operand);
      return;
    }
    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      verifyTransfer(copy);
      return;
    }
    if (auto fill = dyn_cast<linalg::FillOp>(op)) {
      auto p = dict(op, "plan.fill");
      Value value = fill.getOutputs()[0];
      auto type = cast<MemRefType>(value.getType());
      if (string(p, "engine") == "ame") {
        auto consumer = getZeroAccumulatorConsumer(fill);
        auto route = consumer ? dict(consumer, "plan.route") : DictionaryAttr();
        auto init = dict(route, "initialization");
        require(
            op,
            consumer && string(p, "level") == "spm" &&
                string(p, "kind") == "deferred-zero" &&
                string(placement(value), "level") == "spm" &&
                dict(consumer, "contract.task") && dict(op, "contract.task") &&
                integer(p, "consumer") ==
                    integer(consumer ? dict(consumer, "contract.task")
                                     : DictionaryAttr(),
                            "id") &&
                integer(init, "task") ==
                    integer(dict(op, "contract.task"), "id"),
            "deferred zero fill requires a unique complete matmul consumer");
      } else if (string(p, "level") == "host") {
        require(op,
                type.getRank() == 0 && string(p, "engine") == "scalar" &&
                    string(p, "kind") == "init" &&
                    string(placement(value), "level") == "host",
                "scalar fill requires a planned host rank-zero init");
      } else {
        SmallVector<int64_t> strides;
        int64_t rows = type.getRank() == 2 ? type.getDimSize(0) : 1;
        int64_t rowBytes = 0, pitch = 0;
        bool rowSize =
            type.getRank() &&
            !llvm::MulOverflow(type.getShape().back(),
                               bytes(type.getElementType()), rowBytes);
        int64_t vectorBytes = integer(p, "vector_bytes");
        require(op,
                p && string(p, "level") == "spm" &&
                    string(p, "engine") == "rvv" && vectorBytes > 0 &&
                    vectorBytes <= 64 && llvm::isPowerOf2_64(vectorBytes) &&
                    vectorBytes >= bytes(type.getElementType()) &&
                    (string(p, "kind") == "padding" ||
                     string(p, "kind") == "broadcast" ||
                     string(p, "kind") == "overwrite") &&
                    type.getRank() >= 1 && type.getRank() <= 2 &&
                    integer(p, "rows") == rows && rowSize &&
                    integer(p, "row_bytes") == rowBytes && rowBytes > 0 &&
                    rowBytes % vectorBytes == 0 && staticType(value, strides) &&
                    strides.back() == 1 &&
                    (strides.size() == 1 || rows == 1 ||
                     (!llvm::MulOverflow(strides[0],
                                         bytes(type.getElementType()), pitch) &&
                      pitch % 64 == 0)) &&
                    string(placement(value), "level") == "spm" &&
                    aligned(value),
                "unsupported RVV fill shape, layout, or plan");
        bounds(op, value);
      }
      return;
    }
    if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
      verifyMatmul(matmul);
      return;
    }
    if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
      auto p = dict(op, "plan.vector");
      auto mathOperations = p ? p.getAs<ArrayAttr>("math") : ArrayAttr();
      auto operands = p ? p.getAs<ArrayAttr>("operands") : ArrayAttr();
      int64_t elements = integer(p, "elements");
      int64_t lanes = integer(p, "lanes");
      require(op,
              p &&
                  string(p, "engine") ==
                      (mathOperations && !mathOperations.empty() ? "scalar"
                                                                 : "rvv") &&
                  integer(p, "vector_bytes") ==
                      (mathOperations && !mathOperations.empty() ? 4 : 64) &&
                  lanes == pointwiseLanes(generic) && lanes > 0 &&
                  elements > 0 && elements % lanes == 0 &&
                  string(p, "tail") == "none" &&
                  string(p, "fp_policy") ==
                      (mathOperations && !mathOperations.empty()
                           ? "explicit-approximation"
                           : "preserve") &&
                  operands && operands.size() == op->getNumOperands(),
              "missing or unsupported pointwise vector plan");
      if (operands)
        for (auto [i, selection] : llvm::enumerate(operands)) {
          if (i >= op->getNumOperands())
            break;
          binding(op, dyn_cast<DictionaryAttr>(selection), i);
          Value value = op->getOperand(i);
          auto type = dyn_cast<MemRefType>(value.getType());
          AffineMap map = generic.getIndexingMapsArray()[i];
          if (!type) {
            require(op,
                    i < static_cast<size_t>(generic.getNumDpsInputs()) &&
                        supportedPointwiseScalar(value.getType()) &&
                        map.getNumResults() == 0,
                    "pointwise scalar input requires an empty indexing map");
            continue;
          }
          SmallVector<int64_t> strides;
          require(op,
                  type.getRank() == 1 && type.getDimSize(0) == elements &&
                      supportedPointwiseScalar(type.getElementType()) &&
                      bytes(type.getElementType()) > 0 && map.isIdentity() &&
                      staticType(value, strides) && strides[0] == 1,
                  "pointwise memory operands require equal-shaped contiguous "
                  "identity indexing and supported element types");
        }
      Block &body = generic.getRegion().front();
      auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
      auto output =
          generic.getOutputs().size() == 1
              ? dyn_cast<MemRefType>(generic.getOutputs()[0].getType())
              : MemRefType();
      require(
          op,
          output && yield && yield.getValues().size() == 1 &&
              yield.getValues()[0].getType() == output.getElementType() &&
              generic.getNumLoops() == 1 && generic.getNumParallelLoops() == 1,
          "pointwise expression must yield one element with parallel indexing");
      auto scalarOperands =
          p ? p.getAs<ArrayAttr>("scalar_operands") : ArrayAttr();
      unsigned scalarIndex = 0;
      unsigned mathIndex = 0;
      for (auto [operationIndex, arithmetic] : llvm::enumerate(body)) {
        if (!pointwiseMathRecipe(arithmetic).empty()) {
          auto recipe =
              mathOperations && mathIndex < mathOperations.size()
                  ? dyn_cast<DictionaryAttr>(mathOperations[mathIndex])
                  : DictionaryAttr();
          auto abs =
              recipe ? recipe.getAs<FloatAttr>("max_abs_error") : FloatAttr();
          auto rel =
              recipe ? recipe.getAs<FloatAttr>("max_rel_error") : FloatAttr();
          require(
              op,
              supportedPointwiseMath(arithmetic) && recipe &&
                  recipe.size() == 6 &&
                  integer(recipe, "operation") == int64_t(operationIndex) &&
                  string(recipe, "function") ==
                      arithmetic.getName().stripDialect() &&
                  string(recipe, "recipe") == pointwiseMathRecipe(arithmetic) &&
                  string(recipe, "evaluation") ==
                      "scalar-separate-mul-add-rne" &&
                  abs && abs.getType().isF64() &&
                  abs.getValueAsDouble() == pointwiseMathAbsError(arithmetic) &&
                  rel && rel.getType().isF64() &&
                  rel.getValueAsDouble() == pointwiseMathRelError(arithmetic),
              "math recipe must match the operation, accuracy contract "
              "and verified FP32 evaluation");
          ++mathIndex;
        }
        require(op,
                isa<linalg::YieldOp>(arithmetic) ||
                    supportedPointwiseOperation(arithmetic),
                "unsupported lane-wise scalar expression");
        for (auto [index, operand] :
             llvm::enumerate(arithmetic.getOperands())) {
          auto argument = dyn_cast<BlockArgument>(operand);
          if ((argument && argument.getOwner() == &body) ||
              (operand.getDefiningOp() &&
               operand.getDefiningOp()->getBlock() == &body))
            continue;
          auto constant = operand.getDefiningOp<arith::ConstantOp>();
          auto selection =
              scalarOperands && scalarIndex < scalarOperands.size()
                  ? dyn_cast<DictionaryAttr>(scalarOperands[scalarIndex])
                  : DictionaryAttr();
          int64_t recordedOp = selection && selection.get("operation")
                                   ? integer(selection, "operation")
                                   : 0;
          require(op,
                  constant && constant->getBlock() != &body &&
                      isa<IntegerAttr, FloatAttr>(constant.getValue()) &&
                      supportedPointwiseScalar(operand.getType()) &&
                      selection && recordedOp == int64_t(operationIndex) &&
                      integer(selection, "operand") == int64_t(index) &&
                      string(selection, "kind") == "constant-broadcast" &&
                      selection.get("value") == constant.getValue(),
                  "generic scalar capture requires an exact explicit constant "
                  "broadcast plan");
          ++scalarIndex;
        }
      }
      require(op, scalarIndex == (scalarOperands ? scalarOperands.size() : 0),
              "pointwise scalar broadcast plan has extraneous operands");
      require(op,
              (!p || !p.get("math") || mathOperations) &&
                  mathIndex == (mathOperations ? mathOperations.size() : 0),
              "pointwise math plan has missing or extraneous recipes");
      return;
    }
    if (auto reduce = dyn_cast<linalg::ReduceOp>(op)) {
      auto p = dict(op, "plan.reduce");
      Block &body = reduce.getRegion().front();
      Operation *combiner = body.empty() ? nullptr : &body.front();
      bool isSum = isa_and_nonnull<arith::AddFOp>(combiner);
      bool isMaximum = isa_and_nonnull<arith::MaximumFOp>(combiner);
      bool isMaxNum = isa_and_nonnull<arith::MaxNumFOp>(combiner);
      require(
          op,
          string(p, "engine") == "rvv" &&
              ((isSum && integer(p, "vector_bytes") == 64 &&
                string(p, "strategy") == "ordered-vector-add") ||
               (isMaxNum && integer(p, "vector_bytes") == 64 &&
                string(p, "strategy") == "ordered-vector-maxnum") ||
               (isMaximum && integer(p, "vector_bytes") == 4 &&
                string(p, "strategy") == "ordered-scalar-max")) &&
              string(p, "fp_policy") == "preserve" && integer(p, "axis") == 0 &&
              reduce.getDimensions() == ArrayRef<int64_t>{0} &&
              reduce.getInputs().size() == 1 && reduce.getInits().size() == 1,
          "unsupported ordered reduction plan");
      if (reduce.getInputs().size() != 1 || reduce.getInits().size() != 1)
        return;
      binding(op, dict(p, "input"), 0);
      binding(op, dict(p, "output"), 1);
      auto input = cast<MemRefType>(reduce.getInputs()[0].getType());
      auto output = cast<MemRefType>(reduce.getInits()[0].getType());
      SmallVector<int64_t> strides;
      require(
          op,
          input.getRank() == 1 && input.getElementType().isF32() &&
              output.getRank() == 0 && output.getElementType().isF32() &&
              integer(p, "elements") == input.getDimSize(0) &&
              input.getDimSize(0) > 0 && input.getDimSize(0) % 16 == 0 &&
              staticType(reduce.getInputs()[0], strides) && strides[0] == 1 &&
              string(placement(reduce.getInits()[0]), "level") == "host",
          "reduction requires contiguous f32 input and host rank-zero output");
      bool supported =
          combiner &&
          llvm::TypeSwitch<Operation *, bool>(combiner)
              .Case<arith::AddFOp, arith::MaximumFOp, arith::MaxNumFOp>(
                  [](auto op) {
                    return op.getFastmath() == arith::FastMathFlags::none;
                  })
              .Default(false);
      auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
      require(op,
              body.getOperations().size() == 2 && body.getNumArguments() == 2 &&
                  supported && yield && yield.getValues().size() == 1 &&
                  yield.getValues()[0] == combiner->getResult(0) &&
                  ((combiner->getOperand(0) == body.getArgument(0) &&
                    combiner->getOperand(1) == body.getArgument(1)) ||
                   (combiner->getOperand(1) == body.getArgument(0) &&
                    combiner->getOperand(0) == body.getArgument(1))),
              "reduction payload must yield one planned ordered combine of "
              "input and accumulator");
      return;
    }
    if (isa<linalg::LinalgOp>(op))
      require(op, false, "unplanned Linalg operation");
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      auto type = cast<MemRefType>(load.getMemRefType());
      if (auto p = dict(op, "plan.scalar")) {
        SmallVector<int64_t> strides;
        require(op,
                string(p, "engine") == "rvv" && string(p, "kind") == "load" &&
                    integer(p, "elements") == 1 && type.getRank() == 1 &&
                    type.getShape() == ArrayRef<int64_t>{1} &&
                    matchPattern(load.getIndices()[0], m_Zero()) &&
                    staticType(load.getMemRef(), strides) && strides[0] == 1 &&
                    string(placement(load.getMemRef()), "level") == "spm",
                "scalar snapshot requires a complete private SPM singleton");
        binding(op, dict(p, "input"), 0);
      } else
        require(op,
                type.getRank() == 0 &&
                    string(placement(load.getMemRef()), "level") == "host",
                "scalar load must access planned host rank-zero storage");
    }
    if (auto store = dyn_cast<memref::StoreOp>(op))
      require(op,
              cast<MemRefType>(store.getMemRef().getType()).getRank() == 0 &&
                  string(placement(store.getMemRef()), "level") == "host",
              "v1 has no independent external or SPM scalar-store recipe; "
              "use a supported Linalg output and boundary copy");
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      auto p = dict(op, "plan.loop");
      auto cf = dict(op, "contract.cf");
      if (!cf || !cf.getAs<IntegerAttr>("trip_count")) {
        auto graph =
            dict(op->getParentOfType<func::FuncOp>(), "contract.compute_graph");
        auto complete = graph ? graph.getAs<BoolAttr>("complete") : BoolAttr();
        require(op, !p && complete && !complete.getValue(),
                "opaque loop requires an incomplete contract and no window "
                "reuse plan");
        return;
      }
      require(op,
              p && p.getAs<BoolAttr>("spm_window_reuse") &&
                  integer(p, "trips") > 0 &&
                  integer(p, "trips") == integer(cf, "trip_count"),
              "loop plan disagrees with its contract summary");
    }
    if (isa<scf::IfOp>(op)) {
      auto p = dict(op, "plan.scope");
      auto sharing = p ? p.getAs<BoolAttr>("share_spm_windows") : BoolAttr();
      require(op, string(p, "kind") == "exclusive-arms" && sharing,
              "if requires an exclusive-arms scope plan");
    }
  }
  void verifyMatrixTransfer(memref::CopyOp copy, DictionaryAttr p,
                            int64_t count, int64_t size, int64_t sourceStride,
                            int64_t targetStride, bool internal = false) {
    Operation *op = copy;
    auto source = cast<MemRefType>(copy.getSource().getType());
    auto target = cast<MemRefType>(copy.getTarget().getType());
    int64_t element = bytes(source.getElementType());
    bool geometry =
        matrixtransfer::isSupportedElementType(source.getElementType()) &&
        matrixtransfer::getCommonUnitAxis(source, target).has_value() &&
        source.getRank() == 2 && target.getRank() == 2 && size % 4 == 0 &&
        sourceStride % 4 == 0 && targetStride % 4 == 0;
    require(op, geometry,
            "matrix transport requires exact rank-two raw32 rows");
    if (!geometry)
      return;
    int64_t total;
    require(op,
            !llvm::MulOverflow(count, size, total) &&
                integer(p, "total_bytes") == total &&
                integer(p, "temp_bytes") == 0 &&
                string(p, "reason") ==
                    (internal ? "nonoverlapping-spm-exact-acc32-rectangle"
                              : "dma-illegal-exact-acc32-rectangle"),
            "matrix transfer byte accounting or selection reason is invalid");
    for (auto [type, key, pitch] :
         {std::tuple{source, StringRef("source_layout"), sourceStride},
          std::tuple{target, StringRef("target_layout"), targetStride}}) {
      auto layout = dict(p, key);
      SmallVector<int64_t> strides;
      int64_t offset;
      auto shape = layout ? layout.getAs<DenseI64ArrayAttr>("shape")
                          : DenseI64ArrayAttr();
      auto recorded = layout ? layout.getAs<DenseI64ArrayAttr>("strides")
                             : DenseI64ArrayAttr();
      int64_t axis = integer(layout, "contiguous_axis");
      bool exact = succeeded(type.getStridesAndOffset(strides, offset)) &&
                   shape && recorded && shape.asArrayRef() == type.getShape() &&
                   recorded.asArrayRef() == ArrayRef<int64_t>(strides) &&
                   axis >= 0 && axis < 2 && strides[axis] == 1 &&
                   type.getDimSize(axis) == size / element &&
                   type.getDimSize(1 - axis) == count &&
                   integer(layout, "element_bits") == element * 8 &&
                   integer(layout, "descriptor_offset") == offset &&
                   integer(layout, "rows") == count &&
                   integer(layout, "row_bytes") == size &&
                   integer(layout, "row_pitch") == pitch &&
                   string(layout, "origin") == "effective-pointer";
      require(op, exact, "matrix layout evidence disagrees with its endpoint");
    }
    auto tiles = p.getAs<ArrayAttr>("tiles");
    if (!tiles || tiles.empty())
      return;
    require(op, integer(p, "completion") == int64_t(tiles.size()) - 1,
            "matrix completion must name the final tile");
    int64_t row = 0, column = 0;
    for (auto [index, attribute] : llvm::enumerate(tiles)) {
      auto tile = dyn_cast<DictionaryAttr>(attribute);
      int64_t rows = std::min(matrixtransfer::kMaxRows, count - row);
      int64_t width = std::min(matrixtransfer::kMaxRowBytes, size - column);
      bool exact =
          tile && row < count && integer(tile, "id") == int64_t(index) &&
          integer(tile, "predecessor") == int64_t(index) - 1 &&
          integer(tile, "source_offset") == row * sourceStride + column &&
          integer(tile, "target_offset") == row * targetStride + column &&
          integer(tile, "source_pitch") == sourceStride &&
          integer(tile, "target_pitch") == targetStride &&
          integer(tile, "rows") == rows &&
          integer(tile, "row_bytes") == width &&
          integer(tile, "instruction_bits") ==
              matrixtransfer::kInstructionBits &&
          integer(tile, "register_count") == matrixtransfer::kRegisterCount &&
          integer(tile, "mtype") == matrixtransfer::kMtype &&
          string(tile, "register_class") == matrixtransfer::kRegisterClass;
      require(op, exact,
              "matrix tiles must exactly cover the rectangle in dependency "
              "order with the supported ACC32 configuration");
      if (!exact)
        return;
      column += width;
      if (column == size) {
        column = 0;
        row += rows;
      }
    }
    require(op, row == count && column == 0,
            "matrix tiles leave required elements uncovered");
  }
  void verifyInternalTransfer(memref::CopyOp copy, DictionaryAttr p) {
    Operation *op = copy;
    auto source = dyn_cast<MemRefType>(copy.getSource().getType());
    auto target = dyn_cast<MemRefType>(copy.getTarget().getType());
    auto axis = matrixtransfer::getCommonUnitAxis(source, target);
    bool geometry = axis && source.getShape() == target.getShape();
    require(op, geometry,
            "internal copy requires equally shaped supported rank-two "
            "storage with a common physical contiguous axis");
    if (!geometry)
      return;
    for (MemRefType type : {source, target}) {
      SmallVector<int64_t> strides;
      int64_t offset;
      if (failed(type.getStridesAndOffset(strides, offset)) ||
          offset == ShapedType::kDynamic) {
        require(op, false, "internal copy requires static descriptor offsets");
        return;
      }
    }
    auto sourceLayout = *matrixtransfer::getLayout(source, *axis);
    auto targetLayout = *matrixtransfer::getLayout(target, *axis);
    auto chunks = dict(p, "chunks");
    require(op,
            string(p, "role") == "internal" &&
                string(p, "engine") == "matrix" && string(p, "from") == "spm" &&
                string(p, "to") == "spm" &&
                string(p, "recipe") == "spm-acc32-spm" &&
                string(p, "temporary_window") == "none" &&
                string(p, "ordering") == "synchronous-fenced-tiles" &&
                string(p, "wait") == "after-each-tile" &&
                integer(chunks, "count") == sourceLayout.rows &&
                integer(chunks, "bytes") == sourceLayout.rowBytes &&
                integer(chunks, "source_stride") == sourceLayout.pitch &&
                integer(chunks, "target_stride") == targetLayout.pitch,
            "internal copy recipe, effective geometry or completion policy "
            "disagrees with the supported synchronous SPM-ACC32-SPM path");
    SmallVector<std::tuple<int64_t, int64_t, int64_t>> ranges;
    for (auto [index, value] :
         llvm::enumerate(ValueRange{copy.getSource(), copy.getTarget()})) {
      auto selected =
          dict(p, index == 0 ? "source_endpoint" : "target_endpoint");
      auto window = placement(value);
      auto range = byteBounds(value);
      auto origin = spmdma::getStaticPointerOrigin(value);
      auto offset = spmdma::getStaticDescriptorOffset(value);
      int64_t begin, end;
      bool valid =
          privateRoot(root(value)) && string(window, "level") == "spm" &&
          origin && origin->byteShift >= 0 && offset && *offset >= 0 && range &&
          range->first >= 0 && range->second <= integer(window, "capacity") &&
          aligned(value, 4) &&
          !llvm::AddOverflow(integer(window, "offset"), range->first, begin) &&
          !llvm::AddOverflow(integer(window, "offset"), range->second, end);
      require(op, valid,
              "internal copy endpoint requires a static bounded and aligned "
              "private SPM window");
      if (!valid)
        return;
      require(
          op,
          integer(selected, "operand") == int64_t(index) &&
              string(selected, "level") == "spm" &&
              integer(selected, "bank") == integer(window, "bank") &&
              integer(selected, "window_offset") == integer(window, "offset") &&
              integer(selected, "capacity") == integer(window, "capacity") &&
              integer(selected, "begin") == begin &&
              integer(selected, "end") == end,
          "internal copy endpoint evidence disagrees with its physical "
          "SPM placement and full effective range");
      ranges.emplace_back(integer(window, "bank"), begin, end);
    }
    auto [sourceBank, sourceBegin, sourceEnd] = ranges[0];
    auto [targetBank, targetBegin, targetEnd] = ranges[1];
    require(op,
            root(copy.getSource()) != root(copy.getTarget()) &&
                (sourceBank != targetBank || sourceEnd <= targetBegin ||
                 targetEnd <= sourceBegin),
            "internal copy requires distinct nonaliasing roots and "
            "nonoverlapping physical SPM ranges");
    verifyMatrixTransfer(copy, p, sourceLayout.rows, sourceLayout.rowBytes,
                         sourceLayout.pitch, targetLayout.pitch, true);
  }
  void verifyTransfer(memref::CopyOp copy) {
    Operation *op = copy;
    auto p = dict(op, "plan.transfer"), chunks = dict(p, "chunks");
    if (string(p, "role") == "internal" ||
        (string(p, "from") == "spm" && string(p, "to") == "spm")) {
      verifyInternalTransfer(copy, p);
      return;
    }
    bool load = string(p, "from") == "ddr" && string(p, "to") == "spm";
    bool store = string(p, "from") == "spm" && string(p, "to") == "ddr";
    int64_t count = integer(chunks, "count"), size = integer(chunks, "bytes");
    int64_t ddrStride = integer(chunks, "ddr_stride"),
            spmStride = integer(chunks, "spm_stride");
    bool segmented = string(p, "engine") == "segmented";
    bool matrix = string(p, "engine") == "matrix";
    bool legal =
        p && chunks && (load || store) && count > 0 && size > 0 &&
        ddrStride >= 0 && spmStride >= 0 &&
        (count == 1 || (ddrStride >= size && spmStride >= size)) &&
        ((matrix && string(p, "wait") == "after-each-tile") ||
         (segmented && string(p, "wait") == "after-each-segment") ||
         (string(p, "engine") == "dma" &&
          string(p, "wait") == "after-each-chunk" && size % 64 == 0 &&
          (count == 1 || (ddrStride % 64 == 0 && spmStride % 64 == 0))));
    require(op, legal,
            "unsupported DMA direction, chunks, stride, or wait policy");
    if (!legal)
      return;
    Value spm = load ? copy.getTarget() : copy.getSource();
    Value ddr = load ? copy.getSource() : copy.getTarget();
    auto window = placement(spm);
    auto sourceType = dyn_cast<MemRefType>(copy.getSource().getType());
    auto targetType = dyn_cast<MemRefType>(copy.getTarget().getType());
    int64_t element = sourceType ? bytes(sourceType.getElementType()) : 0;
    int64_t alignment = matrix ? 4 : segmented ? element : 64;
    auto ddrPlacement = placement(ddr);
    bool privateDDR = isV2() && privateRoot(root(ddr)) &&
                      string(ddrPlacement, "level") == "ddr";
    if (privateDDR && failed(verifyPrivateEndpoint(op, ddr)))
      invalid = true;
    require(op,
            alignment > 0 && string(window, "level") == "spm" &&
                integer(window, "bank") == integer(p, "channel") &&
                (argumentRoot(root(ddr)) || privateDDR) &&
                aligned(ddr, alignment) && aligned(spm, alignment),
            "DMA endpoints, channel/bank, or alignment disagree with the plan");
    bounds(op, spm);
    bounds(op, ddr);
    SmallVector<int64_t> axes;
    for (auto [value, stride] :
         {std::pair<Value, int64_t>{spm, spmStride}, {ddr, ddrStride}}) {
      auto type = dyn_cast<MemRefType>(value.getType());
      SmallVector<int64_t> strides;
      bool legal = type && staticType(value, strides);
      int64_t width = type ? bytes(type.getElementType()) : 0;
      int64_t axis = -1, extent = 0, pitch = 0, last = 0;
      if (legal && type.getRank() == 1) {
        legal = count == 1 && strides[0] == 1 &&
                !llvm::MulOverflow(type.getDimSize(0), width, extent) &&
                extent == size;
        axis = 0;
      } else if (legal && type.getRank() == 2) {
        for (int minor : {0, 1})
          if (strides[minor] == 1 &&
              !llvm::MulOverflow(type.getDimSize(minor), width, extent) &&
              extent == size && count == type.getDimSize(1 - minor) &&
              !llvm::MulOverflow(strides[1 - minor], width, pitch) &&
              (count == 1 || stride == pitch)) {
            axis = minor;
            break;
          }
        legal = axis >= 0;
      } else {
        legal = false;
      }
      legal = legal && width && size % width == 0 && stride % width == 0 &&
              !llvm::MulOverflow(count - 1, stride, last) &&
              !llvm::AddOverflow(last, size, last);
      require(op, legal && width && size % width == 0 && stride % width == 0,
              "DMA chunks do not exactly cover the operand physical layout");
      axes.push_back(axis);
    }
    bool matching =
        sourceType && targetType &&
        sourceType.getShape() == targetType.getShape() &&
        sourceType.getElementType() == targetType.getElementType() &&
        axes.size() == 2 && axes[0] >= 0 && axes[0] == axes[1];
    require(
        op, matching,
        "DMA cannot change the logical shape or transpose its contiguous axis");
    if (matrix && matching && element) {
      verifyMatrixTransfer(copy, p, count, size, load ? ddrStride : spmStride,
                           load ? spmStride : ddrStride);
      return;
    }
    if (!segmented || !matching || !element)
      return;
    auto segments = p.getAs<ArrayAttr>("segments");
    if (!segments || segments.empty())
      return;
    require(op, integer(p, "completion") == int64_t(segments.size()) - 1,
            "transfer completion must name the final segment");
    auto sourceResidue = spmdma::getEndpointByteRemainder(copy.getSource());
    auto targetResidue = spmdma::getEndpointByteRemainder(copy.getTarget());
    int64_t sourceStride = load ? ddrStride : spmStride;
    int64_t targetStride = load ? spmStride : ddrStride;
    int64_t row = 0, cursor = 0;
    for (auto [index, attribute] : llvm::enumerate(segments)) {
      auto segment = cast<DictionaryAttr>(attribute);
      int64_t srcOffset = integer(segment, "source_offset");
      int64_t dstOffset = integer(segment, "target_offset");
      int64_t length = integer(segment, "bytes");
      int64_t srcBegin = 0, dstBegin = 0, end = 0;
      bool coverage = row < count && length > 0 && srcOffset >= 0 &&
                      dstOffset >= 0 && srcOffset % element == 0 &&
                      dstOffset % element == 0 && length % element == 0 &&
                      !llvm::MulOverflow(row, sourceStride, srcBegin) &&
                      !llvm::AddOverflow(srcBegin, cursor, srcBegin) &&
                      !llvm::MulOverflow(row, targetStride, dstBegin) &&
                      !llvm::AddOverflow(dstBegin, cursor, dstBegin) &&
                      srcOffset == srcBegin && dstOffset == dstBegin &&
                      !llvm::AddOverflow(cursor, length, end) && end <= size;
      require(
          op, coverage,
          "transfer segments must exactly cover each physical row in order");
      if (!coverage)
        return;
      require(op,
              integer(segment, "id") == int64_t(index) &&
                  integer(segment, "predecessor") == int64_t(index) - 1 &&
                  !string(segment, "reason").empty(),
              "transfer segment dependency, id, or reason is invalid");
      StringRef engine = string(segment, "engine");
      if (engine == "dma") {
        require(op,
                length % 64 == 0 && alignedRoot(ddr) && alignedRoot(spm) &&
                    sourceResidue && targetResidue &&
                    (*sourceResidue + srcOffset % 64) % 64 == 0 &&
                    (*targetResidue + dstOffset % 64) % 64 == 0 &&
                    !segment.get("vector_bytes"),
                "DMA segment requires both actual endpoints and length "
                "to be 64-byte aligned");
      } else if (engine == "rvv") {
        int64_t vectorBytes = integer(segment, "vector_bytes");
        require(op,
                element != 2 && vectorBytes > 0 && vectorBytes <= 64 &&
                    vectorBytes % element == 0 && length % vectorBytes == 0 &&
                    aligned(ddr, element) && aligned(spm, element),
                "RVV segment requires naturally aligned exact element vectors");
      } else {
        require(op, false, "unknown transfer segment engine");
      }
      cursor = end;
      if (cursor == size) {
        ++row;
        cursor = 0;
      }
    }
    require(op, row == count && cursor == 0,
            "transfer segments leave required elements uncovered");
  }
  void verifyMatmul(linalg::MatmulOp matmul) {
    Operation *op = matmul;
    if (auto compute = dict(op, "plan.compute")) {
      verifyOrderedMatmul(matmul, compute);
      return;
    }
    auto reference = dict(op, "contract.compute");
    require(op, !reference || !reference.get("order"),
            "an explicit ordered reference computation requires a matching "
            "scalar plan.compute recipe");
    auto p = dict(op, "plan.route"), tile = dict(p, "tile"),
         sub = dict(p, "subtiles");
    int64_t m = integer(tile, "m"), n = integer(tile, "n"),
            k = integer(tile, "k");
    int64_t ns = integer(sub, "n"), ks = integer(sub, "k");
    int64_t fullN = 0, fullK = 0;
    bool geometry = string(p, "engine") == "ame" && m > 0 && m <= 64 && n > 0 &&
                    n <= 64 && k > 0 && k <= 64 && ns > 0 && ks > 0 &&
                    !llvm::MulOverflow(n, ns, fullN) &&
                    !llvm::MulOverflow(k, ks, fullK);
    require(op, geometry, "invalid nr-fpga AME tile or subtile counts");
    if (!geometry || matmul.getNumOperands() != 3)
      return;
    binding(op, dict(p, "a"), 0);
    binding(op, dict(p, "b"), 1);
    binding(op, dict(p, "accumulator"), 2);
    binding(op, dict(p, "output"), 2);
    if (auto init = dict(p, "initialization")) {
      auto proof = getZeroAccumulatorInit(matmul);
      auto fillPlan = proof ? dict(proof->fill, "plan.fill") : DictionaryAttr();
      require(op, bool(proof),
              "zero accumulator source proof requires an exact private zero "
              "fill, complete coverage, and first-iteration lifetime");
      require(op,
              proof && dict(op, "contract.task") &&
                  dict(proof->fill, "contract.task") &&
                  string(init, "kind") ==
                      (proof->loop ? "first-iteration-zero" : "zero") &&
                  integer(init, "task") ==
                      integer(proof ? dict(proof->fill, "contract.task")
                                    : DictionaryAttr(),
                              "id") &&
                  string(fillPlan, "engine") == "ame" &&
                  string(fillPlan, "kind") == "deferred-zero" &&
                  integer(fillPlan, "consumer") ==
                      integer(dict(op, "contract.task"), "id"),
              "zero accumulator initialization and deferred fill recipes "
              "must name the same producer and consumer; a real fill cannot "
              "also be deferred");
    }
    require(op, matmul.getCast() == linalg::TypeFn::cast_signed,
            "AME requires signed i8 multiplication");
    auto a = cast<MemRefType>(matmul.getInputs()[0].getType());
    auto b = cast<MemRefType>(matmul.getInputs()[1].getType());
    auto c = cast<MemRefType>(matmul.getOutputs()[0].getType());
    for (Value input : matmul.getInputs()) {
      bool disjoint = root(input) != root(matmul.getOutputs()[0]);
      if (!disjoint) {
        auto inputRange = byteBounds(input);
        auto outputRange = byteBounds(matmul.getOutputs()[0]);
        disjoint = inputRange && outputRange &&
                   (inputRange->second <= outputRange->first ||
                    outputRange->second <= inputRange->first);
      }
      require(op, disjoint, "AME accumulator may overlap an input operand");
    }
    require(op,
            a.getRank() == 2 && b.getRank() == 2 && c.getRank() == 2 &&
                a.getElementType().isInteger(8) &&
                b.getElementType().isInteger(8) &&
                c.getElementType().isInteger(32) &&
                a.getShape() == ArrayRef<int64_t>{m, fullK} &&
                b.getShape() == ArrayRef<int64_t>{fullK, fullN} &&
                c.getShape() == ArrayRef<int64_t>{m, fullN},
            "AME tile/subtiles disagree with operand shapes or element types");
    AmeTileLayout layouts[] = {AmeTileLayout::MxKWithKContiguous,
                               AmeTileLayout::KxNWithPhysicalNxK,
                               AmeTileLayout::MxNWithNContiguous};
    for (auto [i, value] : llvm::enumerate(matmul.getOperands())) {
      auto type = cast<MemRefType>(value.getType());
      SmallVector<int64_t> strides;
      bool legal =
          staticType(value, strides) && hasAmeTileLayout(type, layouts[i]);
      if (legal) {
        auto layout =
            computeTileStorageLayout(type, i == 0   ? TileRole::A
                                           : i == 1 ? TileRole::B
                                                    : TileRole::Accumulator);
        legal &= layout && layout->rowStrideBytes % 64 == 0;
        if (i == 0 && ks > 1)
          legal &= k * bytes(type.getElementType()) % 64 == 0;
        if (i == 1 && ks > 1)
          legal &= k * bytes(type.getElementType()) % 64 == 0;
        if (i == 2 && ns > 1)
          legal &= n * bytes(type.getElementType()) % 64 == 0;
      }
      require(op, legal, "AME tile addresses or parent pitch are not legal");
    }
  }
  void verifyOrderedMatmul(linalg::MatmulOp matmul, DictionaryAttr compute) {
    Operation *op = matmul;
    if (auto reference = dict(op, "contract.compute");
        reference && reference.get("order"))
      require(op,
              string(reference, "kind") == "matmul" &&
                  string(reference, "order") == string(compute, "order"),
              "scalar compute order disagrees with the neutral reference "
              "execution order");
    require(
        op,
        !op->hasAttr("plan.route") && string(compute, "engine") == "scalar" &&
            string(compute, "kind") == "matmul" &&
            string(compute, "order") == "mnk" &&
            string(compute, "fp_policy") == "preserve",
        "ordered matmul requires one scalar mnk plan preserving the payload");
    auto shape = compute.getAs<DenseI64ArrayAttr>("shape");
    auto operands = compute.getAs<ArrayAttr>("operands");
    require(
        op,
        shape && shape.size() == 3 && operands && operands.size() == 3 &&
            op->getNumOperands() == 3 && matmul.hasPureBufferSemantics(),
        "ordered matmul requires three buffer operands and an [m,n,k] shape");
    if (!shape || shape.size() != 3 || !operands || operands.size() != 3 ||
        op->getNumOperands() != 3 || !matmul.hasPureBufferSemantics())
      return;
    int64_t m = shape[0], n = shape[1], k = shape[2];
    auto a = cast<MemRefType>(matmul.getInputs()[0].getType());
    auto b = cast<MemRefType>(matmul.getInputs()[1].getType());
    auto c = cast<MemRefType>(matmul.getOutputs()[0].getType());
    require(op,
            m > 0 && n > 0 && k > 0 &&
                a.getShape() == ArrayRef<int64_t>{m, k} &&
                b.getShape() == ArrayRef<int64_t>{k, n} &&
                c.getShape() == ArrayRef<int64_t>{m, n},
            "ordered matmul shape disagrees with its static operand shapes");
    for (auto [i, value] : llvm::enumerate(matmul.getOperands())) {
      auto selection = dyn_cast<DictionaryAttr>(operands[i]);
      auto type = cast<MemRefType>(value.getType());
      auto p = placement(value);
      SmallVector<int64_t> strides;
      auto range = byteBounds(value);
      int64_t width = bytes(type.getElementType());
      require(
          op,
          selection && integer(selection, "operand") == int64_t(i) &&
              type.getRank() == 2 && staticType(value, strides) && range &&
              range->first >= 0 && width > 0 && range->first % width == 0,
          "ordered matmul operand requires a static naturally aligned view");
      StringRef level = string(selection, "level");
      if (level == "spm") {
        require(op,
                string(p, "level") == "spm" &&
                    integer(selection, "bank") == integer(p, "bank") &&
                    integer(selection, "offset") == integer(p, "offset"),
                "ordered matmul SPM operand does not match its root placement");
        bounds(op, value);
      } else if (level == "ddr") {
        require(
            op,
            argumentRoot(root(value)) ||
                (isV2() && string(p, "level") == "ddr"),
            "ordered matmul DDR operand must refer to an external argument");
        bounds(op, value);
      } else if (level == "host") {
        auto storage = byteBounds(root(value));
        require(op,
                privateRoot(root(value)) && string(p, "level") == "host" &&
                    range && storage && range->first >= storage->first &&
                    range->second <= storage->second,
                "ordered matmul host operand exceeds its planned private root");
      } else {
        require(op, false, "unsupported ordered matmul operand route");
      }
    }
    Block &body = matmul.getRegion().front();
    auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
    require(op,
            body.getNumArguments() == 3 && yield &&
                yield.getValues().size() == 1,
            "ordered matmul must yield one scalar accumulator");
    for (Operation &payload : body.without_terminator())
      require(op,
              payload.getName().getDialectNamespace() == "arith" &&
                  payload.getNumRegions() == 0 && isMemoryEffectFree(&payload),
              "ordered matmul payload must contain only scalar arithmetic");
  }
  ModuleOp module;
  bool invalid = false;
  DenseMap<Value, DictionaryAttr> placements;
};

struct PlanVerifyPass : PassWrapper<PlanVerifyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanVerifyPass)
  StringRef getArgument() const final { return "bosc-ame-plan-verify"; }
  StringRef getDescription() const final {
    return "Verify concrete BOSCAME physical plans";
  }
  void runOnOperation() final {
    if (failed(verifyBOSCAMEPlan(getOperation())))
      signalPassFailure();
  }
};
} // namespace

LogicalResult buddy::boscame::verifyBOSCAMEPlan(ModuleOp module) {
  return PlanVerifier(module).run();
}
bool buddy::boscame::canShareExclusiveSPMWindow(Value first, Value second) {
  return first && second && first.getDefiningOp() && second.getDefiningOp() &&
         exclusive(first.getDefiningOp(), second.getDefiningOp());
}
std::unique_ptr<Pass> buddy::boscame::createBOSCAMEPlanVerifyPass() {
  return std::make_unique<PlanVerifyPass>();
}
