//===- VIRSchedule.cpp - Order VIR ops for register pressure ----*- C++ -*-===//
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
// -vir-schedule orders the ops of each block of a vir.set_vl region so that
// fewer dynamic vectors are live at once (MaxLive, the Req_Group from which
// -lower-vir-to-vector sizes the vectors). It only moves ops within their
// block: the ops and their operands are unchanged, and the results are bit
// for bit those of the original order.
//
// The order is a post-order of the dependences of the block, from its roots
// (the ops whose results the block does not use, in their order), each op
// placed right before its first user: an operand is computed when it is
// needed, not before. The operands of an op are computed in decreasing order
// of their register need (Sethi-Ullman): the most demanding one first, while
// the fewest values of the other operands are live. An op with regions (a
// loop) is one node, whose need is the peak of its body.
//
// Memory accesses keep their order where it matters: a read stays after the
// write before it, a write after the reads and the write before it. Reads
// move freely among themselves. An op of unknown effects is a write. An op
// that is not speculatable (a division of a divisor not known nonzero, a
// load) stays after the last op before it that may not return (a write, a
// call or an assertion among them, or a loop that may not end): no op runs
// where it did not. Speculatable pure ops move anywhere.
//
// A block is reordered only when its MaxLive, counted as -lower-vir-to-vector
// counts it (IN U OUT at every op, a loop counted at its body's peak), goes
// down. (Every dynamic vector is counted, for any target: on RVV,
// -lower-vir-to-vector does not count a broadcast used only as the scalar
// operand of a .vf instruction.) Each block is scheduled after its nested
// blocks. Linear in the ops of a block, but for the sort of the operands of
// each op.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "VIR/VIRDialect.h"
#include "VIR/VIROps.h"
#include "VIR/VIRTypes.h"

using namespace mlir;
using namespace buddy;

namespace {

bool isDynamicVector(Value v) {
  return isa<vir::DynamicVectorType>(v.getType());
}

/// The dynamic vectors among `values`.
unsigned countDynamicVectors(ValueRange values) {
  return llvm::count_if(values, isDynamicVector);
}

/// How an op orders with the memory accesses of its block.
enum class MemoryKind { None, Read, Write };

MemoryKind memoryKind(Operation *op) {
  if (isMemoryEffectFree(op))
    return MemoryKind::None;
  std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
      getEffectsRecursively(op);
  if (!effects)
    return MemoryKind::Write;
  MemoryKind kind = MemoryKind::None;
  for (const MemoryEffects::EffectInstance &effect : *effects) {
    if (!isa<MemoryEffects::Read>(effect.getEffect()))
      return MemoryKind::Write;
    kind = MemoryKind::Read;
  }
  return kind;
}

/// Whether `op` may not return, which an op after it relies on: it writes
/// (a call, an assertion, an op of unknown effects among them), or it is or
/// holds an op with regions that may not terminate (not speculatable by
/// itself, e.g. an scf.while, an scf.for of a step other than 1).
bool mayNotReturn(Operation *op, MemoryKind kind) {
  if (kind == MemoryKind::Write)
    return true;
  return op
      ->walk([](Operation *nested) {
        if (nested->getNumRegions() == 0)
          return WalkResult::advance();
        auto spec = dyn_cast<ConditionallySpeculatable>(nested);
        if (!spec || spec.getSpeculatability() == Speculation::NotSpeculatable)
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

/// Schedules the blocks of a vir.set_vl region (see above).
class RegionScheduler {
public:
  explicit RegionScheduler(bool verbose) : verbose(verbose) {}

  /// Schedules `block` and the blocks nested in it; returns its MaxLive.
  unsigned schedule(Block &block);

private:
  /// MaxLive of `block` in its current order: at an op, the dynamic vectors
  /// defined in the block (its arguments included) live at it, and, at an
  /// op with regions, the peak of its blocks instead of its results and the
  /// operands that die at it unused inside. The values the block uses from
  /// an enclosing block are counted there, at the op that holds the block.
  unsigned maxLive(Block &block);

  /// The peak of the blocks of `op` (0 without regions).
  unsigned innerPeak(Operation *op);

  bool verbose;
  DenseMap<Block *, unsigned> peakOf;
};

unsigned RegionScheduler::innerPeak(Operation *op) {
  unsigned peak = 0;
  for (Region &region : op->getRegions())
    for (Block &block : region)
      peak = std::max(peak, peakOf.lookup(&block));
  return peak;
}

unsigned RegionScheduler::maxLive(Block &block) {
  DenseMap<Operation *, unsigned> position;
  unsigned n = 0;
  for (Operation &op : block)
    position[&op] = n++;
  if (n == 0)
    return 0;
  // the last position at which `v` is used in the block (`from` if none)
  auto lastUse = [&](Value v, unsigned from) {
    unsigned last = from;
    for (Operation *user : v.getUsers())
      if (Operation *ancestor = block.findAncestorOpInBlock(*user))
        last = std::max(last, position[ancestor]);
    return last;
  };
  // live[p]: the dynamic vectors defined in the block live at position p,
  // by the difference of their starts and ends
  SmallVector<int> delta(n + 1, 0);
  auto addRange = [&](Value v, unsigned start) {
    ++delta[start];
    --delta[lastUse(v, start) + 1];
  };
  for (BlockArgument arg : block.getArguments())
    if (isDynamicVector(arg))
      addRange(arg, 0);
  for (Operation &op : block)
    for (Value result : op.getResults())
      if (isDynamicVector(result))
        addRange(result, position[&op]);
  unsigned peak = 0;
  int live = 0;
  for (Operation &op : block) {
    unsigned p = position[&op];
    live += delta[p];
    unsigned atOp = live;
    if (op.getNumRegions() != 0) {
      // the body's peak, not the results nor the operands dying here
      atOp -= countDynamicVectors(op.getResults());
      SmallPtrSet<Value, 4> seen;
      for (Value operand : op.getOperands()) {
        if (!isDynamicVector(operand) || !seen.insert(operand).second)
          continue;
        Operation *def = operand.getDefiningOp();
        bool local = isa<BlockArgument>(operand)
                         ? cast<BlockArgument>(operand).getOwner() == &block
                         : def && def->getBlock() == &block;
        bool usedInside = llvm::any_of(operand.getUsers(), [&](Operation *u) {
          return op.isProperAncestor(u);
        });
        if (local && !usedInside && lastUse(operand, 0) == p)
          --atOp;
      }
      atOp += innerPeak(&op);
    }
    peak = std::max(peak, atOp);
  }
  return peak;
}

unsigned RegionScheduler::schedule(Block &block) {
  // the nested blocks first: their peaks are the needs of their ops here
  for (Operation &op : block)
    for (Region &region : op.getRegions())
      for (Block &inner : region)
        peakOf[&inner] = schedule(inner);

  SmallVector<Operation *> original;
  for (Operation &op : block.without_terminator())
    original.push_back(&op);
  if (original.size() < 2)
    return peakOf[&block] = maxLive(block);

  // Dependences: the ops of the block whose results an op uses (in its
  // regions too), and the ops it is ordered after: memory accesses, and the
  // last op before it that may not return when it is not speculatable (a
  // division by zero, a load out of bounds must not run before the call
  // that checks it, or a loop that may not end). A barrier (an op that may
  // not return) stays after the writes and the barrier before it.
  DenseMap<Operation *, SmallVector<Operation *, 4>> dataPreds, orderPreds;
  DenseSet<Operation *> hasDataUser;
  Operation *lastWrite = nullptr, *lastBarrier = nullptr;
  SmallVector<Operation *> readsSinceWrite;
  for (Operation *op : original) {
    // an entry for every op: the post-order below only reads them
    orderPreds[op];
    llvm::SetVector<Operation *> preds;
    op->walk([&](Operation *nested) {
      for (Value operand : nested->getOperands())
        if (Operation *def = operand.getDefiningOp();
            def && def->getBlock() == &block)
          preds.insert(def);
    });
    for (Operation *pred : preds)
      hasDataUser.insert(pred);
    dataPreds[op].assign(preds.begin(), preds.end());
    MemoryKind kind = memoryKind(op);
    bool barrier = mayNotReturn(op, kind);
    if (lastBarrier && (barrier || !isSpeculatable(op)))
      orderPreds[op].push_back(lastBarrier);
    if (barrier && kind != MemoryKind::Write && lastWrite)
      orderPreds[op].push_back(lastWrite);
    if (barrier)
      lastBarrier = op;
    switch (kind) {
    case MemoryKind::None:
      break;
    case MemoryKind::Read:
      if (lastWrite)
        orderPreds[op].push_back(lastWrite);
      readsSinceWrite.push_back(op);
      break;
    case MemoryKind::Write:
      if (lastWrite)
        orderPreds[op].push_back(lastWrite);
      orderPreds[op].append(readsSinceWrite);
      readsSinceWrite.clear();
      lastWrite = op;
      break;
    }
  }
  Operation *terminator =
      block.mightHaveTerminator() ? block.getTerminator() : nullptr;
  SmallVector<Operation *, 4> terminatorPreds;
  if (terminator)
    for (Value operand : terminator->getOperands())
      if (Operation *def = operand.getDefiningOp();
          def && def->getBlock() == &block) {
        hasDataUser.insert(def);
        terminatorPreds.push_back(def);
      }

  // need(op): the most dynamic vectors live while it and its operands are
  // computed, its operands in decreasing order of need - results, holding
  // their results; at the op itself, its operands and its results (a loop:
  // its body's peak).
  DenseMap<Operation *, unsigned> need;
  auto results = [](Operation *op) {
    return countDynamicVectors(op->getResults());
  };
  auto sortByNeed = [&](SmallVectorImpl<Operation *> &ops) {
    llvm::stable_sort(ops, [&](Operation *a, Operation *b) {
      return int(need[a]) - int(results(a)) > int(need[b]) - int(results(b));
    });
  };
  for (Operation *op : original) {
    SmallVector<Operation *, 4> preds(dataPreds[op]);
    sortByNeed(preds);
    unsigned held = 0, peak = 0;
    for (Operation *pred : preds) {
      peak = std::max(peak, held + need[pred]);
      held += results(pred);
    }
    unsigned atOp = op->getNumRegions() ? std::max(results(op), innerPeak(op))
                                        : results(op);
    need[op] = std::max(peak, held + atOp);
    // the operands in the order they are computed
    dataPreds[op] = std::move(preds);
  }
  sortByNeed(terminatorPreds);

  // The post-order from the roots, without recursion: the data operands of
  // an op in their order, then the memory accesses it is ordered after.
  SmallVector<Operation *> order;
  DenseSet<Operation *> placed;
  auto place = [&](Operation *root) {
    if (placed.contains(root))
      return;
    SmallVector<std::pair<Operation *, unsigned>> stack;
    stack.push_back({root, 0});
    DenseSet<Operation *> onStack{root};
    while (!stack.empty()) {
      auto &[op, next] = stack.back();
      ArrayRef<Operation *> data = dataPreds.find(op)->second;
      ArrayRef<Operation *> ordered = orderPreds.find(op)->second;
      Operation *pred = nullptr;
      while (!pred && next < data.size() + ordered.size()) {
        Operation *candidate =
            next < data.size() ? data[next] : ordered[next - data.size()];
        ++next;
        if (!placed.contains(candidate))
          pred = candidate;
      }
      if (pred) {
        // the dependences are those of the original order: acyclic
        assert(!onStack.contains(pred) && "cyclic dependences");
        onStack.insert(pred);
        stack.push_back({pred, 0});
        continue;
      }
      placed.insert(op);
      order.push_back(op);
      onStack.erase(op);
      stack.pop_back();
    }
  };
  for (Operation *op : original)
    if (!hasDataUser.contains(op))
      place(op);
  for (Operation *op : terminatorPreds)
    place(op);
  assert(order.size() == original.size() && "an op left unscheduled");

  unsigned before = maxLive(block);
  if (order == ArrayRef<Operation *>(original))
    return peakOf[&block] = before;
  auto reorder = [&](ArrayRef<Operation *> ops) {
    for (Operation *op : ops)
      terminator ? op->moveBefore(terminator)
                 : op->moveBefore(&block, block.end());
  };
  reorder(order);
  unsigned after = maxLive(block);
  if (verbose) {
    llvm::outs() << "vir-schedule: MaxLive " << before << " -> " << after;
    if (after >= before)
      llvm::outs() << ", original order kept";
    llvm::outs() << " in the block of ";
    block.getParentOp()->print(llvm::outs(), OpPrintingFlags().skipRegions());
    llvm::outs() << "\n";
  }
  if (after >= before) {
    reorder(original);
    return peakOf[&block] = before;
  }
  return peakOf[&block] = after;
}

class VIRSchedulePass
    : public PassWrapper<VIRSchedulePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VIRSchedulePass)
  VIRSchedulePass() = default;
  VIRSchedulePass(const VIRSchedulePass &) {}

  StringRef getArgument() const final { return "vir-schedule"; }
  StringRef getDescription() const final {
    return "Order the ops of the blocks of vir.set_vl regions for fewer "
           "dynamic vectors live at once (results unchanged)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vir::VIRDialect>();
  }

  void runOnOperation() override {
    getOperation().walk<WalkOrder::PreOrder>([&](vir::SetVLOp setVL) {
      RegionScheduler scheduler(verbose);
      for (Block &block : setVL.getRegion())
        scheduler.schedule(block);
      return WalkResult::skip();
    });
  }

  Option<bool> verbose{*this, "verbose",
                       llvm::cl::desc("Print MaxLive before and after"),
                       llvm::cl::init(false)};
};
} // namespace

namespace mlir {
namespace buddy {
void registerVIRSchedulePass() { PassRegistration<VIRSchedulePass>(); }
} // namespace buddy
} // namespace mlir
