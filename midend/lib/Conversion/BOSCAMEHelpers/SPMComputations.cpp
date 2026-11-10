//===- SPMComputations.cpp - One-dimensional computation on SPM -----------===//
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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cassert>
#include <optional>

using namespace mlir;
using namespace buddy::boscame::spmdma;

namespace buddy::boscame::lowering {
namespace {

static constexpr StringLiteral kZeroTailSelectPrefixAttr =
    "bosc_ame.zero_tail_select_prefix";
static constexpr StringLiteral kPrefixMaskAuxAttr = "bosc_ame.prefix_mask_aux";
static constexpr StringLiteral kPrefixRangeAttr = "bosc_ame.prefix_range";

static SmallVector<int64_t> graphIds(Operation *op, StringRef name) {
  auto values = op->getAttrOfType<DenseI64ArrayAttr>(name);
  return values ? SmallVector<int64_t>(values.asArrayRef())
                : SmallVector<int64_t>();
}

// Rebuild only user tables that were explicitly supplied. Logical IDs can
// survive bufferization on either tasks, descriptors or function arguments.
static void refreshGraphUsers(func::FuncOp function) {
  Builder builder(function.getContext());
  DenseMap<int64_t, SmallVector<int64_t>> consumers;
  int64_t count = 0;
  function.walk([&](Operation *op) {
    auto id = op->getAttrOfType<IntegerAttr>("bosc_ame.compute_id");
    if (!id)
      return;
    ++count;
    for (int64_t value : graphIds(op, "bosc_ame.compute_input_values")) {
      auto &users = consumers[value];
      if (!llvm::is_contained(users, id.getInt()))
        users.push_back(id.getInt());
    }
  });
  for (auto &entry : consumers)
    llvm::sort(entry.second);
  function.walk([&](Operation *op) {
    auto users = op->getAttrOfType<ArrayAttr>("bosc_ame.logical_value_users");
    auto values =
        op->getAttrOfType<DenseI64ArrayAttr>("bosc_ame.logical_value_ids");
    if (!users || !values || users.size() != static_cast<size_t>(values.size()))
      return;
    SmallVector<Attribute> updated;
    for (int64_t value : values.asArrayRef())
      updated.push_back(builder.getDenseI64ArrayAttr(consumers[value]));
    op->setAttr("bosc_ame.logical_value_users", builder.getArrayAttr(updated));
  });
  for (unsigned i = 0; i < function.getNumArguments(); ++i) {
    auto value =
        function.getArgAttrOfType<IntegerAttr>(i, "bosc_ame.logical_value_id");
    if (value && function.getArgAttrOfType<DenseI64ArrayAttr>(
                     i, "bosc_ame.logical_value_users"))
      function.setArgAttr(
          i, "bosc_ame.logical_value_users",
          builder.getDenseI64ArrayAttr(consumers[value.getInt()]));
  }
  if (function->hasAttr("bosc_ame.compute_graph_node_count"))
    function->setAttr("bosc_ame.compute_graph_node_count",
                      builder.getI64IntegerAttr(count));
}

static bool hasPureParallelPayload(linalg::LinalgOp op) {
  return op.hasPureBufferSemantics() && !op->getNumResults() &&
         op.getNumDpsInits() == 1 &&
         llvm::all_of(op.getIteratorTypesArray(),
                      [](utils::IteratorType type) {
                        return type == utils::IteratorType::parallel;
                      }) &&
         llvm::all_of(
             op->getRegion(0).front().without_terminator(),
             [](Operation &inner) { return isMemoryEffectFree(&inner); });
}

// The static prefix replaces the condition, but both data inputs are still
// read (or inherited for an in-place prefix). Preserve their logical IDs.
static bool dropStaticMaskDependency(linalg::GenericOp op) {
  if (!op->hasAttr(kZeroTailSelectPrefixAttr) ||
      !op->hasAttr("bosc_ame.compute_id"))
    return false;
  auto inputs =
      op->getAttrOfType<DenseI64ArrayAttr>("bosc_ame.compute_input_values");
  // These select operands are all buffers and the unused init is not a read.
  // Partial metadata without this positional mapping must stay conservative.
  if (!inputs || inputs.size() != op.getNumDpsInputs())
    return false;
  auto select = cast<arith::SelectOp>(op.getBody()->front());
  auto condition = cast<BlockArgument>(select.getCondition());
  unsigned position = condition.getArgNumber();
  int64_t maskId = inputs[position];
  SmallVector<int64_t> retained(inputs.asArrayRef());
  retained.erase(retained.begin() + position);
  Builder builder(op.getContext());
  op->setAttr("bosc_ame.compute_input_values",
              builder.getDenseI64ArrayAttr(retained));

  SmallVector<int64_t> predecessors =
      graphIds(op, "bosc_ame.compute_predecessors");
  if (!llvm::is_contained(retained, maskId)) {
    // Drop only a dedicated, pure mask producer. Other ordering edges and
    // producers with additional outputs or barrier semantics remain intact.
    op->getParentOfType<func::FuncOp>().walk([&](Operation *producer) {
      auto task = producer->getAttrOfType<IntegerAttr>("bosc_ame.compute_id");
      auto linalg = dyn_cast<linalg::LinalgOp>(producer);
      auto barrier =
          producer->getAttrOfType<BoolAttr>("bosc_ame.compute_barrier_after");
      auto outputs = graphIds(producer, "bosc_ame.compute_output_values");
      if (!task || !linalg || !producer->hasAttr(kPrefixMaskAuxAttr) ||
          !barrier || barrier.getValue() || !hasPureParallelPayload(linalg) ||
          outputs.size() != 1 || outputs.front() != maskId ||
          linalg.getDpsInits().front() != op.getInputs()[position])
        return;
      llvm::erase_if(predecessors, [&](int64_t predecessor) {
        return predecessor == task.getInt();
      });
    });
  }
  op->setAttr("bosc_ame.compute_predecessors",
              builder.getDenseI64ArrayAttr(predecessors));
  return true;
}

Value index(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

bool touchesSpm(Operation *op) {
  return llvm::any_of(op->getOperandTypes(),
                      [](Type type) { return getSpmBank(type).has_value(); });
}

static bool isDirectExternalStore(linalg::GenericOp op) {
  return op->hasAttr("bosc_ame.direct_ddr_store");
}

// A vector always follows the last physical axis. Outer dimensions are
// scalar loop indices, never extra dimensions of the vector value: rank-two
// vectors are reserved for AME by the common LLVM conversion.
LogicalResult checkVectorBuffer(Operation *op, Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 1 ||
      type.getRank() > 2 ||
      llvm::any_of(type.getShape(), [](int64_t size) { return size <= 0; }))
    return op->emitError(
        "SPM computation needs static positive rank-1 or rank-2 buffers");
  unsigned bytes = getSpmElementByteWidth(type.getElementType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (!bytes || failed(type.getStridesAndOffset(strides, offset)) ||
      strides.back() != 1 ||
      (type.getRank() == 2 && strides.front() < type.getShape().back()))
    return op->emitError(
        "SPM computation needs a contiguous final physical axis");
  if (type.getShape().back() % (kDmaAlignmentBytes / bytes))
    return op->emitError(
        "SPM computation requires complete 64 B vectors; scalar tails are "
        "unsupported");
  return success();
}

int64_t lanesFor(Value value) {
  auto type = cast<MemRefType>(value.getType());
  return kDmaAlignmentBytes / getSpmElementByteWidth(type.getElementType());
}

template <typename Body>
Operation *vectorLoop(OpBuilder &builder, Location loc, MemRefType type,
                      int64_t lanes, Body body) {
  OpBuilder::InsertionGuard guard(builder);
  Value zero = index(builder, loc, 0), one = index(builder, loc, 1);
  SmallVector<Value> indices;
  Operation *outermost = nullptr;
  if (type.getRank() == 2) {
    auto rows = scf::ForOp::create(
        builder, loc, zero, index(builder, loc, type.getDimSize(0)), one);
    builder.setInsertionPointToStart(rows.getBody());
    indices.push_back(rows.getInductionVar());
    outermost = rows;
  }
  auto columns = scf::ForOp::create(builder, loc, zero,
                                    index(builder, loc, type.getShape().back()),
                                    index(builder, loc, lanes));
  builder.setInsertionPointToStart(columns.getBody());
  indices.push_back(columns.getInductionVar());
  body(indices);
  return outermost ? outermost : columns;
}

LogicalResult checkFill(linalg::FillOp op) {
  if (!op.hasPureBufferSemantics() || op.getInputs().size() != 1 ||
      op.getOutputs().size() != 1)
    return op.emitError("SPM fill requires one buffer output");
  return checkVectorBuffer(op, op.getOutputs()[0]);
}

Operation *lowerFill(linalg::FillOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value output = op.getOutputs()[0];
  auto type = cast<MemRefType>(output.getType());
  int64_t lanes = lanesFor(output);
  auto vectorType = VectorType::get({lanes}, type.getElementType());
  Value fill =
      vector::BroadcastOp::create(builder, loc, vectorType, op.getInputs()[0]);
  return vectorLoop(builder, loc, type, lanes, [&](ValueRange indices) {
    vector::StoreOp::create(builder, loc, fill, output, indices);
  });
}

bool isVectorizableBodyOp(Operation &op) {
  // These lower to scalar/vector instructions, without runtime calls. In
  // particular exp/log/tanh/rsqrt cannot silently introduce a libm call while
  // an SPM vector is live. Scalar rsqrt outside this island remains legal.
  return isa<arith::ConstantOp, arith::AddFOp, arith::SubFOp, arith::MulFOp,
             arith::DivFOp, arith::NegFOp, arith::MaximumFOp, arith::MinimumFOp,
             arith::MaxNumFOp, arith::MinNumFOp, arith::AddIOp, arith::SubIOp,
             arith::MulIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
             arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp, arith::CmpFOp,
             arith::CmpIOp, arith::SelectOp, arith::ExtFOp, arith::TruncFOp,
             arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::SIToFPOp,
             arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp, math::AbsFOp,
             math::SqrtOp>(op);
}

bool isVectorElement(Type type) {
  if (type.isF32())
    return true;
  if (auto integer = dyn_cast<IntegerType>(type))
    return llvm::is_contained({1u, 8u, 16u, 32u, 64u}, integer.getWidth());
  // Unsupported float widths and oversized integers may expand into runtime
  // calls even when the MLIR operation itself is side-effect free.
  return false;
}

static bool isPredicateOnlyInput(linalg::GenericOp op, unsigned index) {
  if (index >= op.getNumDpsInputs() || index >= op.getBody()->getNumArguments())
    return false;
  auto type = dyn_cast<MemRefType>(op->getOperand(index).getType());
  if (!type || !type.getElementType().isInteger(1))
    return false;
  BlockArgument argument = op.getBody()->getArgument(index);
  return !argument.use_empty() &&
         llvm::all_of(argument.getUses(), [](OpOperand &use) {
           auto select = dyn_cast<arith::SelectOp>(use.getOwner());
           return select && use.getOperandNumber() == 0;
         });
}

static bool isPredicateBuffer(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || !type.getElementType().isInteger(1) || !type.hasStaticShape() ||
      type.getRank() < 1 || type.getRank() > 2 ||
      llvm::any_of(type.getShape(), [](int64_t size) { return size <= 0; }))
    return false;
  SmallVector<int64_t> strides;
  int64_t offset;
  return succeeded(type.getStridesAndOffset(strides, offset)) &&
         strides.back() == 1;
}

struct StaticIndexPrefixSelect {
  int64_t prefix;
  linalg::IndexOp index;
  arith::CmpIOp compare;
  arith::SelectOp select;
};

static std::optional<StaticIndexPrefixSelect>
matchStaticIndexPrefixSelect(linalg::GenericOp op) {
  if (!op.hasPureBufferSemantics() || op.getOutputs().size() != 1)
    return std::nullopt;
  auto outputType = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
  if (!outputType || outputType.getRank() != 1 || !outputType.hasStaticShape())
    return std::nullopt;

  linalg::IndexOp indexOp;
  arith::CmpIOp compare;
  arith::SelectOp select;
  for (Operation &inner : op.getBody()->without_terminator()) {
    if (isa<linalg::IndexOp>(inner)) {
      if (indexOp)
        return std::nullopt;
      indexOp = cast<linalg::IndexOp>(inner);
    } else if (isa<arith::CmpIOp>(inner)) {
      if (compare)
        return std::nullopt;
      compare = cast<arith::CmpIOp>(inner);
    } else if (isa<arith::SelectOp>(inner)) {
      if (select)
        return std::nullopt;
      select = cast<arith::SelectOp>(inner);
    }
  }
  if (!indexOp || !compare || !select || indexOp.getDim() != 0 ||
      !isa<IndexType>(indexOp.getResult().getType()) ||
      !indexOp.getResult().hasOneUse() ||
      compare.getPredicate() != arith::CmpIPredicate::slt ||
      compare.getLhs() != indexOp.getResult() ||
      !isa<IndexType>(compare.getRhs().getType()) ||
      !compare.getResult().hasOneUse() ||
      select.getCondition() != compare.getResult() ||
      !select.getResult().hasOneUse())
    return std::nullopt;
  auto yield = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
  if (!yield || yield.getValues().size() != 1 ||
      yield.getValues()[0] != select.getResult())
    return std::nullopt;

  auto bound = compare.getRhs().getDefiningOp<arith::ConstantOp>();
  auto boundValue =
      bound ? dyn_cast<IntegerAttr>(bound.getValue()) : IntegerAttr();
  if (!boundValue)
    return std::nullopt;
  int64_t prefix = boundValue.getInt();
  unsigned bytes = getSpmElementByteWidth(outputType.getElementType());
  if (!bytes || bytes > kDmaAlignmentBytes || prefix <= 0 ||
      prefix >= outputType.getDimSize(0) ||
      prefix % (kDmaAlignmentBytes / bytes))
    return std::nullopt;
  return StaticIndexPrefixSelect{prefix, indexOp, compare, select};
}

LogicalResult checkGeneric(linalg::GenericOp op) {
  if (!op.hasPureBufferSemantics() || op.getOutputs().size() != 1 ||
      !llvm::all_of(op.getIteratorTypesArray(), [](utils::IteratorType type) {
        return type == utils::IteratorType::parallel;
      }))
    return op.emitError(
        "SPM generic requires a parallel single-output buffer operation");
  Value output = op.getOutputs()[0];
  auto outputType = cast<MemRefType>(output.getType());
  auto zeroTailPrefix =
      op->getAttrOfType<IntegerAttr>(kZeroTailSelectPrefixAttr);
  if (zeroTailPrefix) {
    unsigned elementBytes = getSpmElementByteWidth(outputType.getElementType());
    if (outputType.getRank() != 1 || !outputType.hasStaticShape() ||
        !elementBytes || elementBytes > kDmaAlignmentBytes ||
        zeroTailPrefix.getInt() <= 0 ||
        zeroTailPrefix.getInt() >= outputType.getDimSize(0) ||
        zeroTailPrefix.getInt() % (kDmaAlignmentBytes / elementBytes) != 0)
      return op.emitError(
          "static zero-tail select requires a positive 64-byte-aligned "
          "prefix on a rank-one buffer");
    Block &body = op.getRegion().front();
    if (body.getOperations().size() != 2)
      return op.emitError("static zero-tail select requires a single select");
    auto select = dyn_cast<arith::SelectOp>(body.front());
    auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
    auto condition = select ? dyn_cast<BlockArgument>(select.getCondition())
                            : BlockArgument();
    auto trueValue = select ? dyn_cast<BlockArgument>(select.getTrueValue())
                            : BlockArgument();
    auto falseValue = select ? dyn_cast<BlockArgument>(select.getFalseValue())
                             : BlockArgument();
    if (!select || !yield || yield.getValues().size() != 1 ||
        yield.getValues()[0] != select.getResult() || !condition ||
        !trueValue || !falseValue ||
        condition.getArgNumber() >= op.getInputs().size() ||
        trueValue.getArgNumber() >= op.getInputs().size() ||
        falseValue.getArgNumber() >= op.getInputs().size())
      return op.emitError("static zero-tail select has an invalid body");
    auto conditionType = dyn_cast<MemRefType>(
        op.getInputs()[condition.getArgNumber()].getType());
    auto trueType = dyn_cast<MemRefType>(
        op.getInputs()[trueValue.getArgNumber()].getType());
    auto falseType = dyn_cast<MemRefType>(
        op.getInputs()[falseValue.getArgNumber()].getType());
    if (!conditionType || !conditionType.getElementType().isInteger(1) ||
        conditionType.getShape() != outputType.getShape() || !trueType ||
        !falseType || trueType.getShape() != outputType.getShape() ||
        falseType.getShape() != outputType.getShape() ||
        trueType.getElementType() != outputType.getElementType() ||
        falseType.getElementType() != outputType.getElementType())
      return op.emitError("static zero-tail select buffers do not match");
  }
  const bool predicateOutput = outputType.getElementType().isInteger(1);
  auto staticIndexSelect = matchStaticIndexPrefixSelect(op);
  if ((predicateOutput && !isPredicateBuffer(output)) ||
      (!predicateOutput && failed(checkVectorBuffer(op, output))))
    return failure();
  const bool directOutput = isDirectExternalStore(op);
  unsigned bytes =
      predicateOutput ? 0 : getSpmElementByteWidth(outputType.getElementType());
  auto maps = op.getIndexingMapsArray();
  for (auto [number, operand] : llvm::enumerate(op.getOperands())) {
    if (auto type = dyn_cast<MemRefType>(operand.getType())) {
      bool isOutput = number >= static_cast<size_t>(op.getNumDpsInputs());
      if (predicateOutput && isOutput) {
        if (!isPredicateBuffer(operand) ||
            type.getShape() != outputType.getShape() ||
            !maps[number].isIdentity())
          return op.emitError(
              "SPM predicate output requires a matching identity layout");
        continue;
      }
      if (isPredicateOnlyInput(op, number)) {
        if (!zeroTailPrefix)
          return op.emitError(
              "SPM predicate select requires a proven static zero-tail "
              "prefix");
        if (type.getShape() != outputType.getShape() ||
            !maps[number].isIdentity())
          return op.emitError(
              "SPM predicate input requires a matching identity layout");
        continue;
      }
      if (!type.getRank()) {
        if (getSpmBank(type))
          return op.emitError("SPM scalar broadcast buffers are unsupported");
        if (maps[number].getNumResults())
          return op.emitError("SPM scalar broadcast requires a rank-zero map");
        continue;
      }
      if (predicateOutput && !type.getElementType().isInteger(1) &&
          getSpmElementByteWidth(type.getElementType()) && !bytes)
        bytes = getSpmElementByteWidth(type.getElementType());
      if (failed(checkVectorBuffer(op, operand)))
        return failure();
      if (!getSpmBank(type) && !(isOutput && directOutput))
        return op.emitError(
            "SPM generic requires placement of every non-scalar operand; "
            "mixed DDR/SPM computation could carry vectors across cache "
            "policy calls");
      if (type.getShape() != outputType.getShape() ||
          !maps[number].isIdentity() ||
          (!predicateOutput &&
           getSpmElementByteWidth(type.getElementType()) != bytes))
        return op.emitError(
            "SPM generic requires matching shapes, element byte widths and "
            "identity indexing maps");
    } else if (!isVectorElement(operand.getType()) ||
               maps[number].getNumResults()) {
      return op.emitError("unsupported SPM generic scalar operand");
    }
  }
  for (Operation &inner : op.getBody()->without_terminator()) {
    if (staticIndexSelect &&
        (&inner == staticIndexSelect->index.getOperation() ||
         &inner == staticIndexSelect->compare.getOperation() ||
         &inner == staticIndexSelect->select.getOperation()))
      continue;
    if (!isVectorizableBodyOp(inner) || inner.getNumRegions() ||
        !llvm::all_of(inner.getResultTypes(), isVectorElement))
      return op.emitError()
             << "unsupported operation in SPM vector computation: "
             << inner.getName();
    if (auto constant = dyn_cast<arith::ConstantOp>(inner))
      if (!isa<IntegerAttr, FloatAttr>(constant.getValue()))
        return op.emitError("SPM vector constants must be scalar numbers");
    for (Value operand : inner.getOperands())
      if (!isVectorElement(operand.getType()))
        return op.emitError(
            "SPM vector computation supports only f32 and native integer "
            "elements");
  }
  if (!predicateOutput && !getSpmBank(outputType) && !directOutput)
    return op.emitError(
        "SPM generic output must be staged in SPM unless it is an explicit "
        "external-DDR store");
  return success();
}

static Operation *lowerZeroTailSelect(linalg::GenericOp op, int64_t prefix) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value output = op.getOutputs()[0];
  auto outputType = cast<MemRefType>(output.getType());
  Block &body = op.getRegion().front();
  auto select = cast<arith::SelectOp>(body.front());
  auto trueArgument = cast<BlockArgument>(select.getTrueValue());
  auto falseArgument = cast<BlockArgument>(select.getFalseValue());
  Value trueInput = op.getInputs()[trueArgument.getArgNumber()];
  Value falseInput = op.getInputs()[falseArgument.getArgNumber()];
  int64_t lanes = lanesFor(output);
  auto vectorType = VectorType::get({lanes}, outputType.getElementType());
  auto copyRange = [&](Value source, int64_t begin, int64_t end) {
    auto forOp = scf::ForOp::create(builder, loc, index(builder, loc, begin),
                                    index(builder, loc, end),
                                    index(builder, loc, lanes));
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(forOp.getBody());
    Value loaded = vector::LoadOp::create(builder, loc, vectorType, source,
                                          ValueRange{forOp.getInductionVar()});
    vector::StoreOp::create(builder, loc, loaded, output,
                            ValueRange{forOp.getInductionVar()});
    return forOp.getOperation();
  };
  // Identity maps and identical generated indices make this an exact
  // self-copy when the source and destination are the same SSA buffer. Keep
  // the tail loop as the task anchor and retain the inherited prefix's
  // logical input dependency in the task metadata.
  if (trueInput != output)
    copyRange(trueInput, 0, prefix);
  return copyRange(falseInput, prefix, outputType.getDimSize(0));
}

static LogicalResult eraseDeadPrefixMaskAuxiliaries(
    ModuleOp module, llvm::function_ref<LogicalResult(ModuleOp)> verifyGraph) {
  bool changed;
  do {
    changed = false;
    SmallVector<Operation *> auxiliaries;
    module.walk([&](Operation *op) {
      if (op->hasAttr(kPrefixMaskAuxAttr) || op->hasAttr(kPrefixRangeAttr))
        auxiliaries.push_back(op);
    });
    for (Operation *op : llvm::reverse(auxiliaries)) {
      auto linalg = dyn_cast<linalg::LinalgOp>(op);
      if (!linalg || !hasPureParallelPayload(linalg))
        continue;
      Value output = linalg.getDpsInits().front();
      Operation *allocation = output.getDefiningOp();
      // A dead local writer is removable; writes to arguments, globals and
      // SPM windows remain observable. Reject every view/escape conservatively.
      if (!allocation || !isa<memref::AllocOp, memref::AllocaOp>(allocation) ||
          allocation->getBlock() != op->getBlock() ||
          allocation->hasAttr("bosc_ame.compute_id") ||
          !cast<MemRefType>(output.getType()).getElementType().isIntOrIndex())
        continue;
      SmallVector<Operation *> deallocations;
      bool live = false;
      for (Operation *user : output.getUsers()) {
        if (user == op)
          continue;
        if (isa<memref::DeallocOp>(user) &&
            !user->hasAttr("bosc_ame.compute_id"))
          deallocations.push_back(user);
        else
          live = true;
      }
      auto function = op->getParentOfType<func::FuncOp>();
      auto task = op->getAttrOfType<IntegerAttr>("bosc_ame.compute_id");
      auto outputs = graphIds(op, "bosc_ame.compute_output_values");
      if (task) {
        auto barrier =
            op->getAttrOfType<BoolAttr>("bosc_ame.compute_barrier_after");
        if (!barrier || barrier.getValue() || outputs.size() != 1)
          continue;
        // Metadata can retain an ordering or logical consumer even when no
        // physical reader remains. Keep those tasks and their incoming edges.
        function.walk([&](Operation *consumer) {
          if (consumer == op)
            return;
          live |= llvm::is_contained(
              graphIds(consumer, "bosc_ame.compute_predecessors"),
              task.getInt());
          for (int64_t value :
               graphIds(consumer, "bosc_ame.compute_input_values"))
            live |= llvm::is_contained(outputs, value);
        });
      }
      if (live)
        continue;
      for (Operation *deallocation : deallocations)
        deallocation->erase();
      op->erase();
      refreshGraphUsers(function);
      if (verifyGraph && failed(verifyGraph(module)))
        return failure();
      changed = true;
    }
  } while (changed);
  return success();
}

Operation *lowerGeneric(linalg::GenericOp op,
                        ArrayRef<Value> registerInputs = {}) {
  if (auto prefix = op->getAttrOfType<IntegerAttr>(kZeroTailSelectPrefixAttr)) {
    return lowerZeroTailSelect(op, prefix.getInt());
  }
  auto staticIndexSelect = matchStaticIndexPrefixSelect(op);
  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value output = op.getOutputs()[0];
  auto outputType = cast<MemRefType>(output.getType());
  int64_t lanes = getSpmElementByteWidth(outputType.getElementType())
                      ? lanesFor(output)
                      : 0;
  if (!lanes)
    for (Value operand : op.getInputs())
      if (auto type = dyn_cast<MemRefType>(operand.getType());
          type && type.getRank() > 0 &&
          getSpmElementByteWidth(type.getElementType())) {
        lanes = lanesFor(operand);
        break;
      }
  assert(lanes > 0 && "predicate vectors need a data operand lane width");
  Block &body = *op.getBody();
  // Read every ordinary scalar before producing any vectors. The late DDR
  // policy pass may introduce calls before these reads, outside the island.
  DenseMap<Value, Value> scalars;
  for (auto [argument, operand] :
       llvm::zip(body.getArguments(), op.getOperands()))
    if (!argument.use_empty()) {
      if (auto type = dyn_cast<MemRefType>(operand.getType())) {
        if (!type.getRank())
          scalars[argument] =
              memref::LoadOp::create(builder, loc, operand, ValueRange{});
      } else {
        scalars[argument] = operand;
      }
    }
  auto emitIteration = [&](ValueRange indices, bool selectTrueBranch) {
    IRMapping mapping;
    // Every input uses these same indices, and all loads precede the payload
    // and its final store. Share exact buffer/type matches only within this
    // iteration's input-loading phase, retaining positional logical inputs.
    DenseMap<std::pair<Value, Type>, Value> inputLoads;
    for (auto [argument, operand] :
         llvm::zip(body.getArguments(), op.getOperands())) {
      if (argument.use_empty())
        continue;
      unsigned position = argument.getArgNumber();
      if (position < registerInputs.size() && registerInputs[position]) {
        mapping.map(argument, registerInputs[position]);
        continue;
      }
      auto vectorType = VectorType::get({lanes}, argument.getType());
      Value value;
      if (auto scalar = scalars.find(argument); scalar != scalars.end())
        value = vector::BroadcastOp::create(builder, loc, vectorType,
                                            scalar->second);
      else {
        Value &loaded = inputLoads[{operand, vectorType}];
        if (!loaded)
          loaded = vector::LoadOp::create(builder, loc, vectorType, operand,
                                          indices);
        value = loaded;
      }
      mapping.map(argument, value);
    }
    auto mapped = [&](Value value) -> Value {
      if (Value result = mapping.lookupOrNull(value))
        return result;
      Value result = vector::BroadcastOp::create(
          builder, loc, VectorType::get({lanes}, value.getType()), value);
      mapping.map(value, result);
      return result;
    };
    for (Operation &inner : body.without_terminator()) {
      if (staticIndexSelect) {
        if (&inner == staticIndexSelect->index.getOperation() ||
            &inner == staticIndexSelect->compare.getOperation())
          continue;
        if (&inner == staticIndexSelect->select.getOperation()) {
          Value selected = selectTrueBranch
                               ? staticIndexSelect->select.getTrueValue()
                               : staticIndexSelect->select.getFalseValue();
          mapping.map(staticIndexSelect->select.getResult(), mapped(selected));
          continue;
        }
      }
      if (auto constant = dyn_cast<arith::ConstantOp>(inner)) {
        auto vectorType = VectorType::get({lanes}, constant.getType());
        Value result = arith::ConstantOp::create(
            builder, inner.getLoc(), vectorType,
            DenseElementsAttr::get(vectorType,
                                   cast<TypedAttr>(constant.getValue())));
        mapping.map(constant.getResult(), result);
        continue;
      }
      OperationState state(inner.getLoc(), inner.getName());
      for (Value operand : inner.getOperands())
        state.addOperands(mapped(operand));
      for (Type type : inner.getResultTypes())
        state.addTypes(VectorType::get({lanes}, type));
      state.addAttributes(inner.getAttrs());
      Operation *created = builder.create(state);
      for (auto [oldResult, newResult] :
           llvm::zip(inner.getResults(), created->getResults()))
        mapping.map(oldResult, newResult);
    }
    auto yield = cast<linalg::YieldOp>(body.getTerminator());
    vector::StoreOp::create(builder, loc, mapped(yield.getValues()[0]), output,
                            indices);
  };

  if (staticIndexSelect) {
    auto emitRange = [&](int64_t begin, int64_t end, bool selectTrueBranch) {
      auto loop = scf::ForOp::create(builder, loc, index(builder, loc, begin),
                                     index(builder, loc, end),
                                     index(builder, loc, lanes));
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      emitIteration(ValueRange{loop.getInductionVar()}, selectTrueBranch);
      return loop.getOperation();
    };
    emitRange(0, staticIndexSelect->prefix, true);
    return emitRange(staticIndexSelect->prefix, outputType.getDimSize(0),
                     false);
  }
  return vectorLoop(builder, loc, outputType, lanes,
                    [&](ValueRange indices) { emitIteration(indices, true); });
}

LogicalResult checkReduce(linalg::ReduceOp op) {
  if (!op.hasPureBufferSemantics() || op.getInputs().size() != 1 ||
      op.getInits().size() != 1 || op.getDimensions().size() != 1 ||
      op.getDimensions()[0] != 0)
    return op.emitError("SPM reduction requires a rank-1 to rank-0 reduction");
  auto input = dyn_cast<MemRefType>(op.getInputs()[0].getType());
  auto output = dyn_cast<MemRefType>(op.getInits()[0].getType());
  if (!input || input.getRank() != 1 || !input.getElementType().isF32() ||
      !output || output.getRank() || !output.getElementType().isF32() ||
      getSpmBank(output))
    return op.emitError(
        "SPM reduction requires an f32 vector and an ordinary scalar result");
  if (failed(checkVectorBuffer(op, op.getInputs()[0])))
    return failure();
  Block &body = op.getRegion().front();
  if (body.getOperations().size() != 2)
    return op.emitError(
        "SPM reduction supports only a single ordered combiner");
  Operation &combiner = body.front();
  bool supported =
      llvm::TypeSwitch<Operation *, bool>(&combiner)
          .Case<arith::AddFOp, arith::MaximumFOp, arith::MaxNumFOp>(
              [](auto op) {
                return op.getFastmath() == arith::FastMathFlags::none;
              })
          .Default(false);
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!supported || !yield || yield.getValues().size() != 1 ||
      yield.getValues()[0] != combiner.getResult(0) ||
      !((combiner.getOperand(0) == body.getArgument(0) &&
         combiner.getOperand(1) == body.getArgument(1)) ||
        (combiner.getOperand(1) == body.getArgument(0) &&
         combiner.getOperand(0) == body.getArgument(1))))
    return op.emitError("SPM reduction requires addf, maximumf or maxnumf "
                        "without fast-math flags");
  return success();
}

Operation *lowerReduce(linalg::ReduceOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value input = op.getInputs()[0], output = op.getInits()[0];
  auto type = cast<MemRefType>(input.getType());
  Block &body = op.getRegion().front();
  const bool isScalarMax = isa<arith::MaximumFOp>(body.front());
  int64_t lanes = isScalarMax ? 1 : lanesFor(input);
  Value initial = memref::LoadOp::create(builder, loc, output, ValueRange{});
  auto loop =
      scf::ForOp::create(builder, loc, index(builder, loc, 0),
                         index(builder, loc, type.getDimSize(0)),
                         index(builder, loc, lanes), ValueRange{initial});
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    Value loaded = vector::LoadOp::create(
        builder, loc, VectorType::get({lanes}, type.getElementType()), input,
        ValueRange{loop.getInductionVar()});
    Value accumulator;
    if (!isScalarMax) {
      // Keep ordered ADD in the IR and use the target's vector reduction.
      // FPGA rounding may differ from per-lane scalar f32 addition; numerical
      // validation therefore uses a tolerance rather than bit identity.
      auto kind = isa<arith::MaxNumFOp>(body.front())
                      ? vector::CombiningKind::MAXNUMF
                      : vector::CombiningKind::ADD;
      accumulator = vector::ReductionOp::create(builder, loc, kind, loaded,
                                                loop.getRegionIterArgs()[0]);
    } else {
      // Preserve the NaN and signed-zero semantics of maximumf. Read through
      // the SPM vector port and apply the original scalar
      // combiner in source order, without vector slides or runtime calls.
      Value element =
          vector::ExtractOp::create(builder, loc, loaded, ArrayRef<int64_t>{0});
      IRMapping mapping;
      mapping.map(body.getArgument(0), element);
      mapping.map(body.getArgument(1), loop.getRegionIterArgs()[0]);
      accumulator = builder.clone(body.front(), mapping)->getResult(0);
    }
    scf::YieldOp::create(builder, loc, accumulator);
  }
  return memref::StoreOp::create(builder, loc, loop.getResult(0), output,
                                 ValueRange{});
}

LogicalResult checkCopy(memref::CopyOp op) {
  if (failed(checkVectorBuffer(op, op.getSource())) ||
      failed(checkVectorBuffer(op, op.getTarget())))
    return failure();
  auto source = cast<MemRefType>(op.getSource().getType());
  auto target = cast<MemRefType>(op.getTarget().getType());
  if (source.getShape() != target.getShape() ||
      source.getElementType() != target.getElementType())
    return op.emitError("SPM copy needs matching static buffer shapes");
  return success();
}

Operation *lowerCopy(memref::CopyOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  auto type = cast<MemRefType>(op.getSource().getType());
  int64_t lanes = lanesFor(op.getSource());
  auto vectorType = VectorType::get({lanes}, type.getElementType());
  return vectorLoop(builder, loc, type, lanes, [&](ValueRange indices) {
    Value loaded = vector::LoadOp::create(builder, loc, vectorType,
                                          op.getSource(), indices);
    vector::StoreOp::create(builder, loc, loaded, op.getTarget(), indices);
  });
}

} // namespace

// Only direct, same-block uses of a private scalar are promoted. Aliases,
// escaping pointers and region/control-flow uses retain their storage.
static bool isPromotableScalar(Value buffer) {
  auto type = dyn_cast<MemRefType>(buffer.getType());
  auto *allocation = buffer.getDefiningOp();
  if (!type || type.getRank() || !type.getElementType().isF32() ||
      getSpmBank(type) ||
      !isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(allocation))
    return false;
  auto hasMetadata = [](Operation *op) {
    return llvm::any_of(op->getAttrs(), [](NamedAttribute attribute) {
      return attribute.getName().getValue().starts_with("bosc_ame.");
    });
  };
  if (hasMetadata(allocation))
    return false;
  for (Operation *user : buffer.getUsers()) {
    if (user->getBlock() != allocation->getBlock())
      return false;
    if (auto store = dyn_cast<memref::StoreOp>(user)) {
      if (!hasMetadata(store))
        continue;
      // Retain task identity on a fresh constant or the immediately preceding
      // reduction loop. Never overwrite an existing task on a shared value.
      Operation *producer = store.getValue().getDefiningOp();
      if (isa_and_nonnull<arith::ConstantOp>(producer))
        continue;
      if (!isa_and_nonnull<scf::ForOp>(producer) ||
          producer->getNumResults() != 1 || producer->getNextNode() != user ||
          hasMetadata(producer))
        return false;
    } else if (auto copy = dyn_cast<memref::CopyOp>(user)) {
      auto source = cast<MemRefType>(copy.getSource().getType());
      auto target = cast<MemRefType>(copy.getTarget().getType());
      if (source.getRank() || target.getRank() || hasMetadata(copy))
        return false;
    } else if (!isa<memref::LoadOp, memref::DeallocOp>(user) ||
               hasMetadata(user)) {
      return false;
    }
  }
  return true;
}

void promotePrivateScalarBuffers(ModuleOp module) {
  DenseMap<Block *, SmallVector<Value>> blocks;
  module.walk([&](Operation *op) {
    if (isa<memref::AllocOp, memref::AllocaOp>(op) &&
        isPromotableScalar(op->getResult(0)))
      blocks[op->getBlock()].push_back(op->getResult(0));
  });
  for (auto &[block, buffers] : blocks) {
    DenseSet<Value> candidates(buffers.begin(), buffers.end());
    DenseSet<Value> initialized, uninitialized;
    for (Operation &op : *block) {
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        if (candidates.contains(load.getMemRef()) && !load.use_empty() &&
            !initialized.contains(load.getMemRef()))
          uninitialized.insert(load.getMemRef());
      } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
        initialized.insert(store.getMemRef());
      } else if (auto copy = dyn_cast<memref::CopyOp>(op)) {
        if (candidates.contains(copy.getSource()) &&
            !initialized.contains(copy.getSource()))
          uninitialized.insert(copy.getSource());
        initialized.insert(copy.getTarget());
      }
    }
    for (Value buffer : uninitialized)
      candidates.erase(buffer);
    DenseMap<Value, Value> values;
    for (Operation &op : llvm::make_early_inc_range(*block)) {
      OpBuilder builder(&op);
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        if (candidates.contains(load.getMemRef())) {
          // Ordinary rank-zero Linalg fills can leave an unused init load
          // before the first write. It does not need an initialized SSA value.
          if (!load.use_empty())
            load.replaceAllUsesWith(values.lookup(load.getMemRef()));
          load.erase();
        }
      } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
        if (!candidates.contains(store.getMemRef()))
          continue;
        Value value = store.getValue();
        bool hasMetadata =
            llvm::any_of(store->getAttrs(), [](NamedAttribute attribute) {
              return attribute.getName().getValue().starts_with("bosc_ame.");
            });
        if (hasMetadata) {
          Operation *anchor = value.getDefiningOp();
          if (auto constant = dyn_cast<arith::ConstantOp>(anchor)) {
            auto clone = arith::ConstantOp::create(builder, store.getLoc(),
                                                   constant.getType(),
                                                   constant.getValue());
            value = clone;
            anchor = clone;
          }
          for (NamedAttribute attribute : store->getAttrs())
            if (attribute.getName().getValue().starts_with("bosc_ame."))
              anchor->setAttr(attribute.getName(), attribute.getValue());
        }
        values[store.getMemRef()] = value;
        store.erase();
      } else if (auto copy = dyn_cast<memref::CopyOp>(op)) {
        bool source = candidates.contains(copy.getSource());
        bool target = candidates.contains(copy.getTarget());
        if (!source && !target)
          continue;
        Value value =
            source ? values.lookup(copy.getSource())
                   : memref::LoadOp::create(builder, copy.getLoc(),
                                            copy.getSource(), ValueRange{});
        if (target)
          values[copy.getTarget()] = value;
        else
          memref::StoreOp::create(builder, copy.getLoc(), value,
                                  copy.getTarget(), ValueRange{});
        copy.erase();
      } else if (auto dealloc = dyn_cast<memref::DeallocOp>(op)) {
        if (candidates.contains(dealloc.getMemref()))
          dealloc.erase();
      }
    }
    for (Value buffer : buffers)
      if (candidates.contains(buffer)) {
        assert(buffer.use_empty() && "scalar promotion left a memory use");
        buffer.getDefiningOp()->erase();
      }
  }
}

LogicalResult checkSpmComputation(Operation *operation) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(operation)
      .Case<linalg::FillOp>(checkFill)
      .Case<linalg::GenericOp>(checkGeneric)
      .Case<linalg::ReduceOp>(checkReduce)
      .Case<memref::CopyOp>(checkCopy)
      .Default([](Operation *op) {
        return op->emitError("unsupported Linalg computation on SPM");
      });
}

LogicalResult
lowerSpmComputations(ModuleOp module,
                     llvm::function_ref<LogicalResult(ModuleOp)> verifyGraph) {
  SpmBroadcastMap broadcasts;
  if (failed(forwardSpmBroadcasts(module, broadcasts, verifyGraph)))
    return failure();
  SmallVector<Operation *> computations;
  WalkResult result = module.walk([&](Operation *op) {
    if (!touchesSpm(op) ||
        (!isa<linalg::LinalgOp>(op) && !isa<memref::CopyOp>(op)))
      return WalkResult::advance();
    if (failed(checkSpmComputation(op)))
      return WalkResult::interrupt();
    computations.push_back(op);
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  for (Operation *op : computations) {
    bool graphChanged = false;
    if (auto generic = dyn_cast<linalg::GenericOp>(op))
      graphChanged = dropStaticMaskDependency(generic);
    Operation *completion =
        llvm::TypeSwitch<Operation *, Operation *>(op)
            .Case<linalg::FillOp>(lowerFill)
            .Case<linalg::GenericOp>([&](auto generic) {
              return lowerGeneric(generic, broadcasts.lookup(generic));
            })
            .Case<linalg::ReduceOp>(lowerReduce)
            .Case<memref::CopyOp>(lowerCopy);
    if (verifyGraph)
      for (NamedAttribute attribute : op->getAttrs())
        if (attribute.getName().getValue().starts_with("bosc_ame."))
          completion->setAttr(attribute.getName(), attribute.getValue());
    auto function = op->getParentOfType<func::FuncOp>();
    op->erase();
    if (verifyGraph && graphChanged)
      refreshGraphUsers(function);
    if (verifyGraph && failed(verifyGraph(module)))
      return failure();
  }
  return eraseDeadPrefixMaskAuxiliaries(module, verifyGraph);
}

void eraseDeadSpmComputationArtifacts(ModuleOp module) {
  // Ordinary Linalg lowering can emit unused loads of output/init values.
  // Use ordinary effect-based DCE for those and dead allocations/constants,
  // while protecting task anchors (including those nested in a dead region).
  bool changed;
  do {
    changed = false;
    module.walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (!isOpTriviallyDead(op))
        return;
      WalkResult tasks = op->walk([](Operation *nested) {
        return nested->hasAttr("bosc_ame.compute_id") ? WalkResult::interrupt()
                                                      : WalkResult::advance();
      });
      if (tasks.wasInterrupted())
        return;
      op->erase();
      changed = true;
    });
  } while (changed);
}

} // namespace buddy::boscame::lowering
