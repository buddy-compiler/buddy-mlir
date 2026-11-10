//===- BOSCAMERegisterSchedule.cpp - Register-resident AME recipes --------===//
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
// Prove, verify, and emit integer AME register-resident schedules. Covered
// contract tasks move to function-level register logical-task tables before
// ordinary buffer materialization.
//
//===----------------------------------------------------------------------===//

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMERegisterSchedule.h"
#include "Analysis/KernelContract.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMETargetInfo.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <functional>
#include <optional>

using namespace mlir;

namespace buddy::boscame {
namespace {

constexpr StringLiteral kSchedule = "plan.register_schedule";
constexpr StringLiteral kMember = "plan.register_member";
constexpr StringLiteral kVirtual = "plan.register_virtual";

Value rootOf(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    auto view = dyn_cast<ViewLikeOpInterface>(op);
    if (!view)
      break;
    value = view.getViewSource();
  }
  return value;
}

bool isPrivate(Value value) {
  return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(
      rootOf(value).getDefiningOp());
}

std::optional<int64_t> field(DictionaryAttr dictionary, StringRef name) {
  auto value = dictionary ? dictionary.getAs<IntegerAttr>(name) : IntegerAttr();
  return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
}

bool zero(Value value) {
  Attribute constant;
  return matchPattern(value, m_Constant(&constant)) &&
         isa<IntegerAttr>(constant) &&
         cast<IntegerAttr>(constant).getInt() == 0;
}

bool positiveMatrix(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.getRank() == 2 && type.hasStaticShape() &&
         type.getDimSize(0) > 0 && type.getDimSize(1) > 0 &&
         !type.getMemorySpace();
}

// A snapshot writes a prefix of the exact private logical descriptor. Root
// equality alone is insufficient: a shifted/reinterpreted alias may refer to
// entirely different elements of the same allocation.
bool privatePrefix(Value prefix, Value logical) {
  if (!positiveMatrix(prefix) || !positiveMatrix(logical) ||
      rootOf(prefix) != rootOf(logical))
    return false;
  auto p = cast<MemRefType>(prefix.getType());
  auto l = cast<MemRefType>(logical.getType());
  SmallVector<int64_t> ps, ls;
  int64_t po, lo;
  auto pOffset = spmdma::getStaticDescriptorOffset(prefix);
  auto lOffset = spmdma::getStaticDescriptorOffset(logical);
  auto pOrigin = spmdma::getStaticPointerOrigin(prefix);
  auto lOrigin = spmdma::getStaticPointerOrigin(logical);
  return p.getElementType() == l.getElementType() && pOffset && lOffset &&
         pOrigin && lOrigin && pOrigin->root == lOrigin->root &&
         pOrigin->byteShift == lOrigin->byteShift && *pOffset == *lOffset &&
         succeeded(p.getStridesAndOffset(ps, po)) &&
         succeeded(l.getStridesAndOffset(ls, lo)) && ps == ls &&
         p.getDimSize(0) <= l.getDimSize(0) &&
         p.getDimSize(1) <= l.getDimSize(1);
}

// A full zero fill is invariant under the physical transpose used to present
// B padding as a contiguous RVV row. Prove an identical dense element set,
// not just equal allocation roots or equal logical element counts.
bool fullZeroFill(linalg::FillOp fill, Value logical) {
  Value output = fill.getOutputs()[0];
  if (!zero(fill.getInputs()[0]) || !positiveMatrix(output) ||
      !positiveMatrix(logical) || rootOf(output) != rootOf(logical))
    return false;
  auto type = cast<MemRefType>(output.getType());
  auto logicalType = cast<MemRefType>(logical.getType());
  auto origin = spmdma::getStaticPointerOrigin(output);
  auto logicalOrigin = spmdma::getStaticPointerOrigin(logical);
  auto offset = spmdma::getStaticDescriptorOffset(output);
  auto logicalOffset = spmdma::getStaticDescriptorOffset(logical);
  auto dense = [](MemRefType type) {
    SmallVector<int64_t> strides;
    int64_t ignored;
    return succeeded(type.getStridesAndOffset(strides, ignored)) &&
           ((strides[1] == 1 && strides[0] == type.getDimSize(1)) ||
            (strides[0] == 1 && strides[1] == type.getDimSize(0)));
  };
  return type.getElementType() == logicalType.getElementType() && origin &&
         logicalOrigin && origin->root == logicalOrigin->root &&
         origin->byteShift == logicalOrigin->byteShift && offset &&
         logicalOffset && *offset == *logicalOffset && dense(type) &&
         dense(logicalType) &&
         type.getNumElements() == logicalType.getNumElements();
}

bool entryRoot(Value root, func::FuncOp function) {
  auto argument = dyn_cast<BlockArgument>(root);
  return argument && argument.getOwner() == &function.getBody().front();
}

bool disjoint(ModuleOp module, Value first, Value second) {
  first = rootOf(first);
  second = rootOf(second);
  if (first == second)
    return false;
  if (isPrivate(first) || isPrivate(second))
    return true;
  auto function = first.getParentRegion()->getParentOfType<func::FuncOp>();
  auto abi = module->getAttrOfType<DictionaryAttr>("contract.abi");
  auto noalias = abi ? abi.getAs<BoolAttr>("args_disjoint") : BoolAttr();
  return function && !function.isExternal() && noalias && noalias.getValue() &&
         entryRoot(first, function) && entryRoot(second, function) &&
         contract::hasEntryArgumentABI(function);
}

bool directEndpoint(ModuleOp module, Value value, unsigned role) {
  if (!positiveMatrix(value) || isPrivate(value))
    return false;
  auto type = cast<MemRefType>(value.getType());
  auto function = value.getParentRegion()->getParentOfType<func::FuncOp>();
  auto abi = module->getAttrOfType<DictionaryAttr>("contract.abi");
  auto alignment =
      abi ? abi.getAs<IntegerAttr>("args_aligned_bytes") : IntegerAttr();
  const auto target = getNrFpgaTargetInfo();
  const auto &capability = role == 0   ? target.ame.a
                           : role == 1 ? target.ame.b
                                       : target.ame.accumulator;
  if (!capability.supportsDirectDdr || !function || function.isExternal() ||
      !entryRoot(rootOf(value), function) ||
      !contract::hasEntryArgumentABI(function) || !alignment ||
      alignment.getInt() < 64 || alignment.getInt() % 64 ||
      !spmdma::isEndpointPointerAlignmentProven(value, 64) ||
      !hasAmeTileLayout(type, capability.layout))
    return false;
  auto layout =
      computeTileStorageLayout(type, role == 0   ? TileRole::A
                                     : role == 1 ? TileRole::B
                                                 : TileRole::Accumulator);
  return layout && layout->rowStrideBytes > 0 &&
         layout->rowStrideBytes % 64 == 0;
}

struct Operand {
  Value logical;
  Value external;
  Value root;
  memref::CopyOp load;
  linalg::FillOp fill;
};

struct Member {
  Operation *operation;
  StringRef role;
};

struct Schedule {
  func::FuncOp function;
  linalg::MatmulOp matmul;
  scf::ForOp loop;
  Operand a, b, c, oldC;
  linalg::GenericOp postAdd;
  memref::CopyOp store;
  Value output;
  Operation *insertion = nullptr;
  SmallVector<Member> members;
  SmallVector<Value> roots;
  int64_t m = 0, n = 0, k = 0, trips = 1;
  int64_t tm = 0, tn = 0, tk = 0, nTiles = 0, kTiles = 0;
  int64_t id = 0;
};

Value initialAccumulator(const Schedule &schedule) {
  return schedule.postAdd ? schedule.oldC.external : schedule.c.external;
}

Operation *inBlock(Operation *op, Block *block) {
  while (op && op->getBlock() != block)
    op = op->getParentOp();
  return op;
}

bool before(Operation *first, Operation *last) {
  Operation *top = inBlock(last, first->getBlock());
  return top && first != top && first->isBeforeInBlock(top);
}

// Enumerate every non-view use, including aliases, so unknown captures and
// metadata users cannot disappear with a virtual allocation.
bool rootUsers(Value root, SmallVectorImpl<Operation *> &users) {
  SmallVector<Value> worklist{root};
  DenseSet<Value> visited;
  DenseSet<Operation *> seen;
  while (!worklist.empty()) {
    Value alias = worklist.pop_back_val();
    if (!visited.insert(alias).second)
      continue;
    for (Operation *user : alias.getUsers()) {
      if (auto view = dyn_cast<ViewLikeOpInterface>(user);
          view && view.getViewSource() == alias && user->getNumResults() == 1) {
        worklist.push_back(user->getResult(0));
      } else if (!isa<memref::DeallocOp>(user) && seen.insert(user).second) {
        users.push_back(user);
      }
    }
  }
  return true;
}

bool inputOperand(Value logical, Operation *consumer, Operand &operand) {
  operand.logical = logical;
  operand.root = rootOf(logical);
  if (!isPrivate(logical)) {
    operand.external = logical;
    return true;
  }
  SmallVector<Operation *> users;
  rootUsers(operand.root, users);
  for (Operation *user : users) {
    if (user == consumer)
      continue;
    if (auto copy = dyn_cast<memref::CopyOp>(user)) {
      if (operand.load || rootOf(copy.getTarget()) != operand.root ||
          !privatePrefix(copy.getTarget(), logical) ||
          isPrivate(copy.getSource()) || !before(copy, consumer))
        return false;
      operand.load = copy;
      operand.external = copy.getSource();
    } else if (auto fill = dyn_cast<linalg::FillOp>(user)) {
      if (operand.fill || !fullZeroFill(fill, logical))
        return false;
      operand.fill = fill;
    } else {
      return false;
    }
  }
  return operand.load && (!operand.fill || before(operand.fill, operand.load));
}

// Recognize the complete integer epilogue before pointwise flattening. The
// named matmul accumulates modulo i32 and this add must do the same: signed
// overflow flags, saturation, floating point and extra region effects cannot
// be reassociated into the initial accumulator.
bool postLoopAdd(Schedule &schedule, linalg::GenericOp add) {
  Value product = schedule.c.logical;
  auto type = cast<MemRefType>(product.getType());
  if (schedule.postAdd || !schedule.loop || !add.hasPureBufferSemantics() ||
      add.getNumDpsInputs() != 2 || add.getNumDpsInits() != 1 ||
      add.getOutputs()[0] != product || add.getNumLoops() != 2 ||
      add->getBlock() != schedule.loop->getBlock() ||
      !schedule.loop->isBeforeInBlock(add) ||
      !llvm::all_of(add.getIteratorTypesArray(),
                    [](utils::IteratorType iterator) {
                      return iterator == utils::IteratorType::parallel;
                    }) ||
      !llvm::all_of(add.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); }))
    return false;
  Block &body = add.getRegion().front();
  if (body.getOperations().size() != 2)
    return false;
  auto sum = dyn_cast<arith::AddIOp>(body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.back());
  if (!sum || !yield || !sum.getType().isInteger(32) ||
      sum.getOverflowFlags() != arith::IntegerOverflowFlags::none ||
      !((sum.getLhs() == body.getArgument(0) &&
         sum.getRhs() == body.getArgument(1)) ||
        (sum.getLhs() == body.getArgument(1) &&
         sum.getRhs() == body.getArgument(0))) ||
      yield.getValues().size() != 1 || yield.getValues()[0] != sum.getResult())
    return false;
  Value old;
  if (add.getInputs()[0] == product)
    old = add.getInputs()[1];
  else if (add.getInputs()[1] == product)
    old = add.getInputs()[0];
  else
    return false;
  auto oldType = dyn_cast<MemRefType>(old.getType());
  if (!positiveMatrix(old) || !isPrivate(old) ||
      oldType.getShape() != type.getShape() ||
      oldType.getElementType() != type.getElementType() ||
      rootOf(old) == schedule.c.root ||
      !inputOperand(old, add, schedule.oldC) ||
      !before(schedule.loop, schedule.oldC.load))
    return false;
  schedule.postAdd = add;
  return true;
}

bool accumulatorOperand(Schedule &schedule) {
  Operand &operand = schedule.c;
  operand.logical = schedule.matmul.getOutputs()[0];
  operand.root = rootOf(operand.logical);
  if (!isPrivate(operand.logical)) {
    operand.external = operand.logical;
    schedule.output = operand.logical;
    return true;
  }
  SmallVector<Operation *> users;
  rootUsers(operand.root, users);
  for (Operation *user : users) {
    if (user == schedule.matmul)
      continue;
    if (auto copy = dyn_cast<memref::CopyOp>(user)) {
      if (rootOf(copy.getTarget()) == operand.root) {
        if (operand.load || !privatePrefix(copy.getTarget(), operand.logical) ||
            isPrivate(copy.getSource()) || !before(copy, schedule.matmul))
          return false;
        operand.load = copy;
        operand.external = copy.getSource();
      } else if (rootOf(copy.getSource()) == operand.root) {
        if (schedule.store ||
            !privatePrefix(copy.getSource(), operand.logical) ||
            isPrivate(copy.getTarget()))
          return false;
        schedule.store = copy;
        schedule.output = copy.getTarget();
      } else {
        return false;
      }
    } else if (auto fill = dyn_cast<linalg::FillOp>(user)) {
      if (operand.fill || !fullZeroFill(fill, operand.logical) ||
          !before(fill, schedule.matmul))
        return false;
      operand.fill = fill;
    } else if (auto add = dyn_cast<linalg::GenericOp>(user)) {
      if (!postLoopAdd(schedule, add))
        return false;
    } else {
      return false;
    }
  }
  if (!schedule.store || (!operand.load && !operand.fill) ||
      (operand.load && operand.fill && !before(operand.fill, operand.load)))
    return false;
  // Replacing (zero + products) + oldC by oldC + products is an integer
  // modulo-i32 identity. It is valid only for the fully zeroed product and
  // the unique terminal add/store proven here, not an arbitrary initial P.
  if (schedule.postAdd && (operand.load || !operand.fill ||
                           !before(schedule.postAdd, schedule.store)))
    return false;
  return true;
}

void addMembers(Schedule &schedule, const Operand &operand, StringRef loadRole,
                StringRef fillRole) {
  if (operand.load)
    schedule.members.push_back({operand.load, loadRole});
  if (operand.fill)
    schedule.members.push_back({operand.fill, fillRole});
  if (isPrivate(operand.logical))
    schedule.roots.push_back(operand.root);
}

// All effects between the original snapshots and output completion must be
// covered. This conservative proof permits moving reads to their tile uses
// and keeping all C versions in registers without hiding an observation,
// source mutation, call, or runtime clobber in the lifetime.
bool closedLifetime(Schedule &schedule) {
  auto *block = schedule.insertion->getBlock();
  Operation *start = schedule.insertion;
  DenseSet<Operation *> members;
  DenseSet<Value> roots(schedule.roots.begin(), schedule.roots.end());
  for (const Member &member : schedule.members) {
    members.insert(member.operation);
    Operation *top = inBlock(member.operation, block);
    if (!top || (top != schedule.insertion &&
                 !top->isBeforeInBlock(schedule.insertion)))
      return false;
    if (top->isBeforeInBlock(start))
      start = top;
  }
  std::function<bool(Operation *)> safe = [&](Operation *op) {
    if (members.contains(op) || isa<scf::YieldOp>(op))
      return true;
    if (op == schedule.loop) {
      for (Operation &nested : schedule.loop.getBody()->getOperations())
        if (!safe(&nested))
          return false;
      return true;
    }
    if (isa<memref::AllocOp, memref::AllocaOp>(op))
      return roots.contains(op->getResult(0));
    if (auto dealloc = dyn_cast<memref::DeallocOp>(op))
      return roots.contains(rootOf(dealloc.getMemref()));
    return op->getNumRegions() == 0 && isMemoryEffectFree(op);
  };
  for (Operation *op = start;; op = op->getNextNode()) {
    if (!op || !safe(op))
      return false;
    if (op == schedule.insertion)
      break;
  }
  return true;
}

// Named linalg operations may carry explicit indexing maps and a serialized
// scalar region. Reconstruct the arithmetic as well as trusting the op name:
// this recipe implements only the standard signed-i8 modulo-i32 reduction.
bool canonicalIntegerMatmul(linalg::MatmulOp matmul) {
  MLIRContext *context = matmul.getContext();
  auto m = getAffineDimExpr(0, context);
  auto n = getAffineDimExpr(1, context);
  auto k = getAffineDimExpr(2, context);
  SmallVector<AffineMap> maps{AffineMap::get(3, 0, {m, k}, context),
                              AffineMap::get(3, 0, {k, n}, context),
                              AffineMap::get(3, 0, {m, n}, context)};
  if (matmul.getIndexingMapsArray() != maps ||
      matmul.getIteratorTypesArray() !=
          SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                           utils::IteratorType::parallel,
                                           utils::IteratorType::reduction})
    return false;
  Block &body = matmul.getRegion().front();
  if (body.getNumArguments() != 3 || body.getOperations().size() != 5)
    return false;
  auto operation = body.begin();
  auto a = dyn_cast<arith::ExtSIOp>(&*operation++);
  auto b = dyn_cast<arith::ExtSIOp>(&*operation++);
  auto product = dyn_cast<arith::MulIOp>(&*operation++);
  auto sum = dyn_cast<arith::AddIOp>(&*operation++);
  auto yield = dyn_cast<linalg::YieldOp>(&*operation);
  return a && b && product && sum && yield &&
         a.getIn() == body.getArgument(0) && b.getIn() == body.getArgument(1) &&
         a.getType().isInteger(32) && b.getType().isInteger(32) &&
         product.getLhs() == a.getOut() && product.getRhs() == b.getOut() &&
         product.getOverflowFlags() == arith::IntegerOverflowFlags::none &&
         sum.getLhs() == body.getArgument(2) &&
         sum.getRhs() == product.getResult() &&
         sum.getOverflowFlags() == arith::IntegerOverflowFlags::none &&
         yield.getValues().size() == 1 &&
         yield.getValues()[0] == sum.getResult();
}

std::optional<Schedule> reconstruct(linalg::MatmulOp matmul) {
  auto module = matmul->getParentOfType<ModuleOp>();
  const auto target = getNrFpgaTargetInfo();
  if (matmul->hasAttr("plan.stream") || matmul->hasAttr("plan.stream_member") ||
      matmul.getCast() != linalg::TypeFn::cast_signed ||
      matmul->hasAttr("contract.compute") || matmul.getNumResults() ||
      !canonicalIntegerMatmul(matmul) ||
      !positiveMatrix(matmul.getInputs()[0]) ||
      !positiveMatrix(matmul.getInputs()[1]) ||
      !positiveMatrix(matmul.getOutputs()[0]))
    return std::nullopt;
  auto aType = cast<MemRefType>(matmul.getInputs()[0].getType());
  auto bType = cast<MemRefType>(matmul.getInputs()[1].getType());
  auto cType = cast<MemRefType>(matmul.getOutputs()[0].getType());
  if (!aType.getElementType().isInteger(8) ||
      !bType.getElementType().isInteger(8) ||
      !cType.getElementType().isInteger(32) ||
      !hasAmeTileLayout(aType, target.ame.a.layout) ||
      !hasAmeTileLayout(bType, target.ame.b.layout) ||
      !hasAmeTileLayout(cType, target.ame.accumulator.layout) ||
      aType.getDimSize(0) != cType.getDimSize(0) ||
      bType.getDimSize(1) != cType.getDimSize(1) ||
      aType.getDimSize(1) != bType.getDimSize(0))
    return std::nullopt;

  Schedule schedule;
  schedule.function = matmul->getParentOfType<func::FuncOp>();
  schedule.matmul = matmul;
  schedule.loop = dyn_cast<scf::ForOp>(matmul->getParentOp());
  if (schedule.loop) {
    auto lower = getConstantIntValue(schedule.loop.getLowerBound());
    auto upper = getConstantIntValue(schedule.loop.getUpperBound());
    auto step = getConstantIntValue(schedule.loop.getStep());
    int64_t distance, rounded;
    if (schedule.loop.getNumResults() || schedule.loop.getNumRegionIterArgs() ||
        !lower || !upper || !step || *step <= 0 || *upper <= *lower ||
        llvm::SubOverflow(*upper, *lower, distance) ||
        llvm::AddOverflow(distance, *step - 1, rounded))
      return std::nullopt;
    schedule.trips = rounded / *step;
  }
  if (!inputOperand(matmul.getInputs()[0], matmul, schedule.a) ||
      !inputOperand(matmul.getInputs()[1], matmul, schedule.b) ||
      !accumulatorOperand(schedule))
    return std::nullopt;
  if (schedule.loop && ((isPrivate(schedule.c.logical) &&
                         schedule.c.root.getDefiningOp()->getBlock() ==
                             schedule.loop.getBody()) ||
                        (schedule.c.load && schedule.c.load->getBlock() ==
                                                schedule.loop.getBody()) ||
                        (schedule.c.fill && schedule.c.fill->getBlock() ==
                                                schedule.loop.getBody())))
    return std::nullopt;

  schedule.insertion = schedule.store  ? schedule.store.getOperation()
                       : schedule.loop ? schedule.loop.getOperation()
                                       : matmul.getOperation();
  if (schedule.store) {
    Operation *compute =
        schedule.loop ? schedule.loop.getOperation() : matmul.getOperation();
    if (compute->getBlock() != schedule.store->getBlock() ||
        !compute->isBeforeInBlock(schedule.store))
      return std::nullopt;
  }
  Value initial = initialAccumulator(schedule);
  if (!directEndpoint(module, schedule.a.external, 0) ||
      !directEndpoint(module, schedule.b.external, 1) ||
      !directEndpoint(module, schedule.output, 2) ||
      (initial && !directEndpoint(module, initial, 2)) ||
      !disjoint(module, schedule.output, schedule.a.external) ||
      !disjoint(module, schedule.output, schedule.b.external))
    return std::nullopt;

  auto outputType = cast<MemRefType>(schedule.output.getType());
  auto sourceA = cast<MemRefType>(schedule.a.external.getType());
  auto sourceB = cast<MemRefType>(schedule.b.external.getType());
  schedule.m = outputType.getDimSize(0);
  schedule.n = outputType.getDimSize(1);
  schedule.k = aType.getDimSize(1);
  if (schedule.m > cType.getDimSize(0) || schedule.n > cType.getDimSize(1) ||
      sourceA.getDimSize(0) < schedule.m ||
      sourceA.getDimSize(1) != schedule.k ||
      sourceB.getDimSize(0) != schedule.k ||
      sourceB.getDimSize(1) < schedule.n ||
      (initial &&
       (cast<MemRefType>(initial.getType()).getDimSize(0) < schedule.m ||
        cast<MemRefType>(initial.getType()).getDimSize(1) < schedule.n)))
    return std::nullopt;
  // Shrinking the independent M/N axes is valid only for the private result
  // observed through this one prefix store. Contract masks alone are not a
  // license to change the original mathematical computation.
  schedule.tm = schedule.m;
  schedule.tn = std::min(schedule.n, target.ame.maxTileN);
  schedule.tk = std::min(schedule.k, target.ame.maxTileK);
  schedule.nTiles = llvm::divideCeil(schedule.n, schedule.tn);
  schedule.kTiles = llvm::divideCeil(schedule.k, schedule.tk);
  if (schedule.m > target.ame.maxTileM ||
      schedule.nTiles > target.allocatableAccumulatorRegisters ||
      target.allocatableTileRegisters < 2 ||
      schedule.kTiles > 1024 / schedule.nTiles)
    return std::nullopt;

  addMembers(schedule, schedule.a, "a-load", "a-padding");
  addMembers(schedule, schedule.b, "b-load", "b-padding");
  addMembers(schedule, schedule.c, "c-load",
             schedule.postAdd  ? "product-zero"
             : schedule.c.load ? "c-padding"
                               : "c-zero");
  if (schedule.postAdd) {
    addMembers(schedule, schedule.oldC, "c-load", "c-padding");
    schedule.members.push_back({schedule.postAdd, "c-post-add"});
  }
  schedule.members.push_back({matmul, "compute"});
  if (schedule.store)
    schedule.members.push_back({schedule.store, "c-store"});
  DenseSet<Operation *> uniqueMembers;
  DenseSet<Value> uniqueRoots;
  for (const Member &member : schedule.members)
    if (!uniqueMembers.insert(member.operation).second ||
        member.operation->hasAttr("plan.stream") ||
        member.operation->hasAttr("plan.stream_member"))
      return std::nullopt;
  for (Value root : schedule.roots)
    if (!uniqueRoots.insert(root).second ||
        root.getDefiningOp()->hasAttr("plan.stream_virtual"))
      return std::nullopt;
  if (!closedLifetime(schedule))
    return std::nullopt;

  DominanceInfo dominance(module);
  for (Value value : {initial, schedule.output})
    if (value && !dominance.dominates(value, schedule.insertion))
      return std::nullopt;
  for (Value value : {schedule.a.external, schedule.b.external}) {
    Operation *definition = value.getDefiningOp();
    bool nested =
        definition && schedule.loop && schedule.loop->isAncestor(definition);
    if (!nested && !dominance.dominates(value, schedule.insertion))
      return std::nullopt;
  }
  return schedule;
}

DictionaryAttr recipe(const Schedule &schedule, Builder &builder) {
  const auto target = getNrFpgaTargetInfo();
  NamedAttrList values;
  auto number = [&](StringRef name, int64_t value) {
    values.set(name, builder.getI64IntegerAttr(value));
  };
  number("version", 1);
  number("id", schedule.id);
  number("m", schedule.m);
  number("n", schedule.n);
  number("k", schedule.k);
  number("tile_m", schedule.tm);
  number("tile_n", schedule.tn);
  number("tile_k", schedule.tk);
  number("n_tiles", schedule.nTiles);
  number("k_tiles", schedule.kTiles);
  number("outer_k_trips", schedule.trips);
  number("peak_tr", 2);
  number("peak_acc", schedule.nTiles);
  number("available_tr", target.allocatableTileRegisters);
  number("available_acc", target.allocatableAccumulatorRegisters);
  values.set("kind", builder.getStringAttr("integer-ame-register"));
  values.set("order", builder.getStringAttr("outer-k-k-n"));
  values.set(
      "initialization",
      builder.getStringAttr(initialAccumulator(schedule) ? "ddr" : "zero"));
  values.set("epilogue",
             builder.getStringAttr(schedule.postAdd ? "post-loop-modulo-i32-add"
                                                    : "none"));
  values.set("carry_accumulators", builder.getBoolAttr(bool(schedule.loop)));
  return values.getDictionary(builder.getContext());
}

DictionaryAttr ownership(int64_t id, StringRef role, Builder &builder) {
  NamedAttrList values;
  values.set("schedule", builder.getI64IntegerAttr(id));
  if (!role.empty())
    values.set("role", builder.getStringAttr(role));
  return values.getDictionary(builder.getContext());
}

DictionaryAttr archive(Operation *operation, Builder &builder) {
  NamedAttrList values;
  for (StringRef name : {"task", "provenance", "access", "fill"})
    if (Attribute attribute = operation->getAttr(("contract." + name).str()))
      values.set(name, attribute);
  return values.getDictionary(builder.getContext());
}

void appendArray(Operation *operation, StringRef name,
                 ArrayRef<Attribute> added, Builder &builder) {
  SmallVector<Attribute> values;
  if (auto previous = operation->getAttrOfType<ArrayAttr>(name))
    llvm::append_range(values, previous);
  llvm::append_range(values, added);
  operation->setAttr(name, builder.getArrayAttr(values));
}

Value tileView(OpBuilder &builder, Location loc, Value source, int64_t row,
               int64_t column, int64_t rows, int64_t columns) {
  return memref::SubViewOp::create(
      builder, loc, source,
      ArrayRef<OpFoldResult>{builder.getIndexAttr(row),
                             builder.getIndexAttr(column)},
      ArrayRef<OpFoldResult>{builder.getIndexAttr(rows),
                             builder.getIndexAttr(columns)},
      ArrayRef<OpFoldResult>{builder.getIndexAttr(1), builder.getIndexAttr(1)});
}

void direct(Operation *operation, OpBuilder &builder) {
  operation->setAttr("bosc_ame.direct_ddr", builder.getUnitAttr());
}

// Only address/scalar expressions in the original source loop are cloned.
// The lifetime proof has already excluded calls and observable side effects.
Value cloneAddress(Value value, scf::ForOp loop, IRMapping &mapping,
                   OpBuilder &builder) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;
  Operation *definition = value.getDefiningOp();
  if (!definition || !loop || !loop->isAncestor(definition))
    return value;
  for (Value operand : definition->getOperands())
    mapping.map(operand, cloneAddress(operand, loop, mapping, builder));
  Operation *cloned = builder.clone(*definition, mapping);
  for (NamedAttribute attr : llvm::make_early_inc_range(cloned->getAttrs()))
    if (attr.getName().getValue().starts_with("contract.") ||
        attr.getName().getValue().starts_with("plan."))
      cloned->removeAttr(attr.getName());
  return mapping.lookup(value);
}

SmallVector<Value> emitStep(const Schedule &schedule, Value a, Value b,
                            ValueRange initial, OpBuilder &builder,
                            Location loc) {
  SmallVector<Value> accumulators(initial);
  for (int64_t ki = 0; ki < schedule.kTiles; ++ki) {
    int64_t k = std::min(schedule.tk, schedule.k - ki * schedule.tk);
    Value aView = tileView(builder, loc, a, 0, ki * schedule.tk, schedule.m, k);
    auto lhs = TileLoadAOp::create(
        builder, loc, VectorType::get({schedule.m, k}, builder.getI8Type()),
        aView);
    direct(lhs, builder);
    // A survives until every output tile has consumed it. B's preceding
    // value dies at each MMA, leaving two TR values simultaneously live.
    for (int64_t ni = 0; ni < schedule.nTiles; ++ni) {
      int64_t n = std::min(schedule.tn, schedule.n - ni * schedule.tn);
      Value bView =
          tileView(builder, loc, b, ki * schedule.tk, ni * schedule.tn, k, n);
      auto rhs = TileLoadBOp::create(
          builder, loc, VectorType::get({k, n}, builder.getI8Type()), bView);
      direct(rhs, builder);
      auto mma = TileMmaOp::create(
          builder, loc, VectorType::get({schedule.m, n}, builder.getI32Type()),
          accumulators[ni], lhs.getMatrix(), rhs.getMatrix());
      mma->setAttr("bosc_ame.ordered", builder.getUnitAttr());
      accumulators[ni] = mma.getMatrix();
    }
  }
  return accumulators;
}

void removeVirtualRoot(Value root) {
  SmallVector<Value> worklist{root};
  SmallVector<Operation *> views;
  SmallVector<Operation *> deallocations;
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value alias = worklist.pop_back_val();
    if (!visited.insert(alias).second)
      continue;
    for (Operation *user : alias.getUsers()) {
      if (isa<memref::DeallocOp>(user))
        deallocations.push_back(user);
      else if (isa<ViewLikeOpInterface>(user)) {
        views.push_back(user);
        worklist.push_back(user->getResult(0));
      }
    }
  }
  for (Operation *deallocation : deallocations)
    deallocation->erase();
  for (Operation *view : llvm::reverse(views))
    if (view->use_empty())
      view->erase();
  if (root.use_empty())
    root.getDefiningOp()->erase();
}

LogicalResult emitSchedule(Schedule schedule, int64_t executionId) {
  OpBuilder builder(schedule.insertion);
  auto loc = schedule.matmul.getLoc();
  auto group = scf::ExecuteRegionOp::create(builder, loc, TypeRange{});
  group.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&group.getRegion().front());
  SmallVector<Value> accumulators;
  for (int64_t ni = 0; ni < schedule.nTiles; ++ni) {
    int64_t n = std::min(schedule.tn, schedule.n - ni * schedule.tn);
    auto type = VectorType::get({schedule.m, n}, builder.getI32Type());
    if (Value initial = initialAccumulator(schedule)) {
      Value view =
          tileView(builder, loc, initial, 0, ni * schedule.tn, schedule.m, n);
      auto loaded = TileLoadAccOp::create(builder, loc, type, view);
      direct(loaded, builder);
      accumulators.push_back(loaded.getMatrix());
    } else {
      accumulators.push_back(
          TileZeroAccOp::create(builder, loc, type).getMatrix());
    }
  }
  if (schedule.loop) {
    auto loop = scf::ForOp::create(builder, loc, schedule.loop.getLowerBound(),
                                   schedule.loop.getUpperBound(),
                                   schedule.loop.getStep(), accumulators);
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      IRMapping mapping;
      mapping.map(schedule.loop.getInductionVar(), loop.getInductionVar());
      Value a =
          cloneAddress(schedule.a.external, schedule.loop, mapping, builder);
      Value b =
          cloneAddress(schedule.b.external, schedule.loop, mapping, builder);
      auto next =
          emitStep(schedule, a, b, loop.getRegionIterArgs(), builder, loc);
      scf::YieldOp::create(builder, loc, next);
    }
    accumulators.assign(loop.getResults().begin(), loop.getResults().end());
  } else {
    accumulators = emitStep(schedule, schedule.a.external, schedule.b.external,
                            accumulators, builder, loc);
  }
  for (int64_t ni = 0; ni < schedule.nTiles; ++ni) {
    int64_t n = std::min(schedule.tn, schedule.n - ni * schedule.tn);
    Value view = tileView(builder, loc, schedule.output, 0, ni * schedule.tn,
                          schedule.m, n);
    auto stored = TileStoreAccOp::create(builder, loc, accumulators[ni], view);
    direct(stored, builder);
  }
  scf::YieldOp::create(builder, loc);

  auto function = group->getParentOfType<func::FuncOp>();
  SmallVector<Attribute> tasks, mappings;
  for (const Member &member : schedule.members) {
    auto task =
        member.operation->getAttrOfType<DictionaryAttr>("contract.task");
    if (!task)
      continue;
    tasks.push_back(archive(member.operation, builder));
    NamedAttrList mapping;
    bool eliminated =
        member.role.ends_with("padding") || member.role == "product-zero";
    mapping.set("logical_task_id", task.getAs<IntegerAttr>("id"));
    mapping.set(
        "execution_ids",
        builder.getDenseI64ArrayAttr(
            eliminated ? ArrayRef<int64_t>{} : ArrayRef<int64_t>{executionId}));
    mapping.set("action",
                builder.getStringAttr(eliminated ? "eliminated" : "executed"));
    mappings.push_back(mapping.getDictionary(builder.getContext()));
  }
  group->setAttr("bosc_ame.execution_id",
                 builder.getI64IntegerAttr(executionId));
  group->setAttr("bosc_ame.register_tasks", builder.getArrayAttr(tasks));
  group->setAttr("bosc_ame.register_schedule", recipe(schedule, builder));
  if (schedule.loop) {
    NamedAttrList logicalLoop;
    for (NamedAttribute attribute : schedule.loop->getAttrs())
      if (attribute.getName().getValue().starts_with("contract."))
        logicalLoop.set(attribute.getName(), attribute.getValue());
    group->setAttr("bosc_ame.register_logical_loop",
                   logicalLoop.getDictionary(builder.getContext()));
  }
  appendArray(function, "bosc_ame.register_logical_tasks", tasks, builder);
  appendArray(function, "bosc_ame.register_task_map", mappings, builder);

  SmallVector<Value> outsideRoots;
  for (Value root : schedule.roots)
    if (!schedule.loop || !schedule.loop->isAncestor(root.getDefiningOp()))
      outsideRoots.push_back(root);
  for (const Member &member : schedule.members)
    if (!schedule.loop || !schedule.loop->isAncestor(member.operation))
      member.operation->erase();
  if (schedule.loop)
    schedule.loop->erase();
  for (Value root : outsideRoots)
    removeVirtualRoot(root);
  return success();
}

} // namespace

bool isRegisterScheduleMember(Operation *operation) {
  return operation && operation->hasAttr(kMember);
}

bool isRegisterScheduleVirtual(Operation *operation) {
  return operation && operation->hasAttr(kVirtual);
}

LogicalResult analyzeRegisterSchedules(ModuleOp module,
                                       ArrayRef<Value> fixedRoots) {
  Builder builder(module.getContext());
  int64_t nextId = 0;
  module.walk([&](linalg::MatmulOp matmul) {
    if (auto id = field(matmul->getAttrOfType<DictionaryAttr>(kSchedule), "id"))
      nextId = std::max(nextId, *id + 1);
  });
  WalkResult result = module.walk([&](linalg::MatmulOp matmul) {
    auto candidate = reconstruct(matmul);
    auto previous = matmul->getAttrOfType<DictionaryAttr>(kSchedule);
    if (candidate && llvm::any_of(candidate->roots, [&](Value root) {
          return llvm::any_of(
              fixedRoots, [&](Value fixed) { return rootOf(fixed) == root; });
        }))
      candidate.reset();
    if (!candidate) {
      if (previous) {
        matmul.emitError("register schedule lost its closed direct-DDR proof");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    candidate->id = field(previous, "id").value_or(nextId++);
    matmul->setAttr(kSchedule, recipe(*candidate, builder));
    for (const Member &member : candidate->members)
      member.operation->setAttr(kMember,
                                ownership(candidate->id, member.role, builder));
    for (Value root : candidate->roots)
      root.getDefiningOp()->setAttr(kVirtual,
                                    ownership(candidate->id, "", builder));
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult verifyRegisterSchedules(ModuleOp module) {
  Builder builder(module.getContext());
  DenseSet<Operation *> expectedMembers, expectedRoots, expectedSchedules;
  DenseSet<int64_t> ids;
  bool invalid = false;
  module.walk([&](Operation *operation) {
    if (!operation->hasAttr(kSchedule))
      return;
    auto matmul = dyn_cast<linalg::MatmulOp>(operation);
    auto recorded = operation->getAttrOfType<DictionaryAttr>(kSchedule);
    auto id = field(recorded, "id");
    auto candidate = matmul ? reconstruct(matmul) : std::nullopt;
    if (!candidate || !id || *id < 0 || !ids.insert(*id).second) {
      operation->emitError("invalid register schedule anchor or ownership");
      invalid = true;
      return;
    }
    candidate->id = *id;
    expectedSchedules.insert(operation);
    if (recorded != recipe(*candidate, builder)) {
      operation->emitError("register schedule does not match its proven graph");
      invalid = true;
    }
    for (const Member &member : candidate->members)
      if (!expectedMembers.insert(member.operation).second ||
          member.operation->getAttr(kMember) !=
              ownership(*id, member.role, builder)) {
        member.operation->emitError(
            "invalid register schedule task membership");
        invalid = true;
      }
    for (Value root : candidate->roots)
      if (!expectedRoots.insert(root.getDefiningOp()).second ||
          root.getDefiningOp()->getAttr(kVirtual) !=
              ownership(*id, "", builder)) {
        root.getDefiningOp()->emitError(
            "invalid register virtual-root ownership");
        invalid = true;
      }
  });
  module.walk([&](Operation *operation) {
    if ((operation->hasAttr(kMember) && !expectedMembers.contains(operation)) ||
        (operation->hasAttr(kVirtual) && !expectedRoots.contains(operation)) ||
        (operation->hasAttr(kSchedule) &&
         !expectedSchedules.contains(operation))) {
      operation->emitError("orphaned register schedule decision");
      invalid = true;
    }
  });
  return failure(invalid);
}

LogicalResult emitRegisterSchedules(ModuleOp module) {
  if (failed(verifyRegisterSchedules(module)))
    return failure();
  SmallVector<Schedule, 4> schedules;
  module.walk([&](linalg::MatmulOp matmul) {
    auto recorded = matmul->getAttrOfType<DictionaryAttr>(kSchedule);
    if (!recorded)
      return;
    Schedule schedule = *reconstruct(matmul);
    schedule.id = *field(recorded, "id");
    schedules.push_back(std::move(schedule));
  });
  for (auto function : module.getOps<func::FuncOp>()) {
    int64_t nextExecution = 0;
    function.walk([&](Operation *operation) {
      if (auto id =
              operation->getAttrOfType<IntegerAttr>("bosc_ame.execution_id"))
        nextExecution = std::max(nextExecution, id.getInt() + 1);
    });
    for (const Schedule &schedule : schedules)
      if (schedule.function == function &&
          failed(emitSchedule(schedule, nextExecution++)))
        return failure();
  }
  return success();
}

} // namespace buddy::boscame
