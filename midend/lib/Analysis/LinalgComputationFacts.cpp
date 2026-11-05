//===- LinalgComputationFacts.cpp - Explicit Linalg SSA facts -------------===//
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
// Builds a read-only LinalgComputationFacts snapshot for one ModuleOp.
//
// What is recorded
// ----------------
// * Preorder operation indices and first-occurrence value indices over block
//   arguments, operands, and results.
// * Structural Linalg kinds (elementwise / reduction / matmul / fill / other)
//   from iterator types and op classes — not hardware recipes.
// * Optional memory-event facts for memref.copy, linalg.fill, and annotated
//   one-in/one-out generics carrying bosc_ame.memory.
// * Reduction reducer graphs, neutral elements, and seed-vs-neutral identity
//   when they are directly readable from IR; never inferred by reassociation.
//
// What is intentionally not inferred
// ----------------------------------
// Control-flow joins, alias sets, buffer contents, execution schedules, or
// SPM/DDR placement legality. Memref SSA values are descriptors only; aliases
// and content versions stay distinct. Reanalyze after any IR rewrite.
//
//===----------------------------------------------------------------------===//

#include "Analysis/LinalgComputationFacts.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include <utility>

using namespace mlir;

namespace buddy::boscame {

// Classify from op class / iterator types only. An all-parallel generic may
// still permute layout; that remains Elementwise here, not a transpose proof.
LinalgComputationKind classifyLinalgComputation(linalg::LinalgOp operation) {
  if (isa<linalg::MatmulOp>(operation.getOperation()))
    return LinalgComputationKind::Matmul;
  if (isa<linalg::FillOp>(operation.getOperation()))
    return LinalgComputationKind::Fill;
  if (isa<linalg::ReduceOp>(operation.getOperation()))
    return LinalgComputationKind::Reduction;
  if (isa<linalg::GenericOp>(operation.getOperation())) {
    auto iterators = operation.getIteratorTypesArray();
    if (llvm::is_contained(iterators, utils::IteratorType::reduction))
      return LinalgComputationKind::Reduction;
    if (llvm::all_of(iterators, [](utils::IteratorType iterator) {
          return iterator == utils::IteratorType::parallel;
        }))
      return LinalgComputationKind::Elementwise;
  }
  return LinalgComputationKind::Other;
}

static std::optional<int64_t> getNonnegativeId(Operation *operation,
                                               StringRef name) {
  auto attribute = operation->getAttrOfType<IntegerAttr>(name);
  if (!attribute || attribute.getInt() < 0)
    return std::nullopt;
  return attribute.getInt();
}

static std::optional<int64_t> getOriginId(Operation *operation) {
  std::optional<int64_t> origin;
  bool conflicting = false;
  auto merge = [&](std::optional<int64_t> candidate) {
    if (!candidate)
      return;
    if (origin && origin != candidate)
      conflicting = true;
    else
      origin = candidate;
  };
  merge(getNonnegativeId(operation, "bosc_ame.origin_id"));
  merge(getNonnegativeId(operation, "bosc_ame.compute_origin_id"));
  if (auto memory = operation->getAttrOfType<DictionaryAttr>("bosc_ame.memory"))
    if (auto id = memory.getAs<IntegerAttr>("origin_id"))
      if (id.getInt() >= 0)
        merge(id.getInt());
  return conflicting ? std::nullopt : origin;
}

static std::optional<TypedAttr> getTensorFillConstant(Value initValue) {
  auto tensorType = dyn_cast<TensorType>(initValue.getType());
  auto fill = initValue.getDefiningOp<linalg::FillOp>();
  if (!tensorType || !fill || !fill.hasPureTensorSemantics() ||
      fill.getNumResults() != 1 || fill.getDpsInits().size() != 1 ||
      fill.getInputs().size() != 1)
    return std::nullopt;

  auto constant = fill.getInputs().front().getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto value = dyn_cast<TypedAttr>(constant.getValue());
  if (!value || value.getType() != tensorType.getElementType())
    return std::nullopt;
  return value;
}

static bool isNaNNeutralWithNoNaNFlag(Operation *combiner, TypedAttr neutral) {
  auto floatNeutral = dyn_cast<FloatAttr>(neutral);
  if (!floatNeutral || !floatNeutral.getValue().isNaN())
    return false;
  auto fastMath = dyn_cast<arith::ArithFastMathInterface>(combiner);
  return fastMath &&
         arith::bitEnumContainsAny(fastMath.getFastMathFlagsAttr().getValue(),
                                   arith::FastMathFlags::nnan);
}

static std::optional<TypedAttr> getScalarNeutralElement(Operation *combiner) {
  if (combiner->getNumResults() != 1 ||
      !isa<IntegerType, FloatType>(combiner->getResult(0).getType()))
    return std::nullopt;
  auto neutral = arith::getNeutralElement(combiner);
  if (!neutral || isNaNNeutralWithNoNaNFlag(combiner, *neutral))
    return std::nullopt;
  return neutral;
}

static std::optional<LinalgMemoryEventFact>
getLinalgMemoryEventFact(Operation *operation,
                         const LinalgOperationFact &operationFact) {
  LinalgMemoryEventFact event;
  if (operation->getName().getStringRef() == "memref.copy") {
    auto memory = operation->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
    if (!memory)
      return std::nullopt;
    event.kind = LinalgMemoryEventKind::Copy;
    event.originId = operationFact.originId;
    if (auto kind = memory.getAs<StringAttr>("kind")) {
      if (kind.getValue() == "load") {
        event.role = LinalgMemoryEventRole::Load;
        event.effect = LinalgMemoryEventEffect::Read;
      } else if (kind.getValue() == "store") {
        event.role = LinalgMemoryEventRole::Store;
        event.effect = LinalgMemoryEventEffect::Write;
      }
    }
    if (operationFact.operandValueIndices.size() >= 2) {
      event.inputValueIndices.push_back(operationFact.operandValueIndices[0]);
      event.outputValueIndices.push_back(operationFact.operandValueIndices[1]);
    }
    return event;
  }

  auto fill = dyn_cast<linalg::FillOp>(operation);
  if (fill) {
    event.kind = LinalgMemoryEventKind::Fill;
    event.role = operation->hasAttr("bosc_ame.padding")
                     ? LinalgMemoryEventRole::Padding
                     : LinalgMemoryEventRole::Logical;
    event.effect = LinalgMemoryEventEffect::Write;
    event.originId = operationFact.originId;
    for (OpOperand *input : fill.getDpsInputOperands())
      if (input->getOperandNumber() < operationFact.operandValueIndices.size())
        event.inputValueIndices.push_back(
            operationFact.operandValueIndices[input->getOperandNumber()]);
    for (OpOperand &output : fill.getDpsInitsMutable())
      if (output.getOperandNumber() < operationFact.operandValueIndices.size())
        event.outputValueIndices.push_back(
            operationFact.operandValueIndices[output.getOperandNumber()]);
    return event;
  }

  // Annotated memory copies are normalized to a one-input/one-output generic
  // before the buffer-form planner runs. Preserve the external event's read
  // or write endpoint as a typed association to that generic's DPS operand.
  auto generic = dyn_cast<linalg::GenericOp>(operation);
  auto memory = operation->getAttrOfType<DictionaryAttr>("bosc_ame.memory");
  auto kind = memory ? memory.getAs<StringAttr>("kind") : StringAttr();
  if (!generic || !kind || generic.getNumDpsInputs() != 1 ||
      generic.getNumDpsInits() != 1)
    return std::nullopt;
  if (kind.getValue() == "load") {
    event.role = LinalgMemoryEventRole::Load;
    event.effect = LinalgMemoryEventEffect::Read;
  } else if (kind.getValue() == "store") {
    event.role = LinalgMemoryEventRole::Store;
    event.effect = LinalgMemoryEventEffect::Write;
  } else {
    return std::nullopt;
  }
  event.kind = LinalgMemoryEventKind::Copy;
  event.originId = operationFact.originId;
  for (OpOperand *input : generic.getDpsInputOperands())
    if (input->getOperandNumber() < operationFact.operandValueIndices.size())
      event.inputValueIndices.push_back(
          operationFact.operandValueIndices[input->getOperandNumber()]);
  for (OpOperand &output : generic.getDpsInitsMutable())
    if (output.getOperandNumber() < operationFact.operandValueIndices.size())
      event.outputValueIndices.push_back(
          operationFact.operandValueIndices[output.getOperandNumber()]);
  return event;
}

LinalgComputationFacts analyzeLinalgComputationFacts(ModuleOp module) {
  LinalgComputationFacts facts;
  module.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    facts.operationIndices.try_emplace(operation, facts.operations.size());
    LinalgOperationFact fact;
    fact.operation = operation;
    facts.operations.push_back(std::move(fact));
  });

  auto getValueIndex = [&](Value value) {
    auto found = facts.valueIndices.find(value);
    if (found != facts.valueIndices.end())
      return found->second;

    size_t index = facts.values.size();
    facts.valueIndices.try_emplace(value, index);
    LinalgValueFact fact;
    fact.value = value;
    fact.type = value.getType();
    fact.elementType = fact.type;
    if (auto shaped = dyn_cast<ShapedType>(fact.type)) {
      fact.isShaped = true;
      fact.ranked = shaped.hasRank();
      fact.elementType = shaped.getElementType();
      if (fact.ranked)
        llvm::append_range(fact.shape, shaped.getShape());
    }

    if (auto result = dyn_cast<OpResult>(value)) {
      Operation *producer = result.getOwner();
      auto producerIndex = facts.operationIndices.find(producer);
      if (producerIndex != facts.operationIndices.end())
        fact.producerOperationIndex = producerIndex->second;
      fact.resultNumber = result.getResultNumber();
      auto logicalIds = producer->getAttrOfType<DenseI64ArrayAttr>(
          "bosc_ame.logical_value_ids");
      if (logicalIds && logicalIds.size() == producer->getNumResults()) {
        int64_t id = logicalIds[result.getResultNumber()];
        if (id >= 0)
          fact.frontendLogicalValueId = id;
      }
    } else if (auto argument = dyn_cast<BlockArgument>(value)) {
      Block *block = argument.getOwner();
      if (Region *region = block->getParent()) {
        Operation *owner = region->getParentOp();
        auto ownerIndex = facts.operationIndices.find(owner);
        if (ownerIndex != facts.operationIndices.end())
          fact.blockOwnerOperationIndex = ownerIndex->second;
        fact.regionNumber = region->getRegionNumber();
        fact.blockNumber = block->computeBlockNumber();
        fact.argumentNumber = argument.getArgNumber();
        if (auto function = dyn_cast<func::FuncOp>(owner))
          if (!function.getBody().empty() &&
              block == &function.getBody().front())
            if (auto id = function.getArgAttrOfType<IntegerAttr>(
                    argument.getArgNumber(), "bosc_ame.logical_value_id"))
              if (id.getInt() >= 0)
                fact.frontendLogicalValueId = id.getInt();
      }
    }
    facts.values.push_back(std::move(fact));
    return index;
  };

  for (auto [operationIndex, fact] : llvm::enumerate(facts.operations)) {
    Operation *operation = fact.operation;
    auto parent = facts.operationIndices.find(operation->getParentOp());
    if (parent != facts.operationIndices.end())
      fact.parentOperationIndex = parent->second;
    auto function = dyn_cast<func::FuncOp>(operation);
    if (!function)
      function = operation->getParentOfType<func::FuncOp>();
    if (function) {
      auto functionIndex = facts.operationIndices.find(function.getOperation());
      if (functionIndex != facts.operationIndices.end())
        fact.functionOperationIndex = functionIndex->second;
    }

    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          getValueIndex(argument);
    for (OpOperand &operand : operation->getOpOperands()) {
      size_t valueIndex = getValueIndex(operand.get());
      fact.operandValueIndices.push_back(valueIndex);
      facts.values[valueIndex].uses.push_back(
          {operationIndex, operand.getOperandNumber()});
    }
    for (Value result : operation->getResults())
      fact.resultValueIndices.push_back(getValueIndex(result));

    fact.computeId = getNonnegativeId(operation, "bosc_ame.compute_id");
    fact.originId = getOriginId(operation);
    fact.memoryEvent = getLinalgMemoryEventFact(operation, fact);
    auto linalg = dyn_cast<linalg::LinalgOp>(operation);
    if (!linalg)
      continue;
    LinalgComputationFact computation;
    computation.kind = classifyLinalgComputation(linalg);
    for (OpOperand *input : linalg.getDpsInputOperands())
      computation.dpsInputOperandNumbers.push_back(input->getOperandNumber());
    for (OpOperand &init : linalg.getDpsInitsMutable()) {
      computation.dpsInitOperandNumbers.push_back(init.getOperandNumber());
      computation.initPayloadReads.push_back(
          linalg.payloadUsesValueFromOperand(&init));
    }
    llvm::append_range(computation.indexingMaps, linalg.getIndexingMapsArray());
    llvm::append_range(computation.iteratorTypes,
                       linalg.getIteratorTypesArray());
    if (computation.kind == LinalgComputationKind::Reduction) {
      if (auto reduce = dyn_cast<linalg::ReduceOp>(operation)) {
        llvm::append_range(computation.reductionDimensions,
                           reduce.getDimensions());
      } else {
        for (auto [dimension, iterator] :
             llvm::enumerate(computation.iteratorTypes))
          if (iterator == utils::IteratorType::reduction)
            computation.reductionDimensions.push_back(dimension);
      }
      LinalgReductionStructureFact reduction;
      for (unsigned operandNumber : computation.dpsInitOperandNumbers)
        if (operandNumber < fact.operandValueIndices.size())
          reduction.initOperandValueIndices.push_back(
              fact.operandValueIndices[operandNumber]);

      if (operation->getNumRegions() == 1) {
        Region &region = operation->getRegion(0);
        if (region.hasOneBlock()) {
          Block &block = region.front();
          for (BlockArgument argument : linalg.getRegionInputArgs())
            reduction.reducerInputArgumentValueIndices.push_back(
                getValueIndex(argument));
          for (BlockArgument argument : linalg.getRegionOutputArgs())
            reduction.reducerInitArgumentValueIndices.push_back(
                getValueIndex(argument));
          if (auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator())) {
            for (Value yieldedValue : yield.getValues())
              reduction.yieldedValueIndices.push_back(
                  getValueIndex(yieldedValue));
            reduction.reducerStructureComplete =
                reduction.reducerInputArgumentValueIndices.size() ==
                    computation.dpsInputOperandNumbers.size() &&
                reduction.reducerInitArgumentValueIndices.size() ==
                    computation.dpsInitOperandNumbers.size() &&
                reduction.initOperandValueIndices.size() ==
                    computation.dpsInitOperandNumbers.size() &&
                reduction.yieldedValueIndices.size() ==
                    computation.dpsInitOperandNumbers.size();
          }
          region.walk<WalkOrder::PreOrder>([&](Operation *payloadOperation) {
            if (payloadOperation == block.getTerminator())
              return;
            auto payloadIndex = facts.operationIndices.find(payloadOperation);
            if (payloadIndex != facts.operationIndices.end())
              reduction.reducerBodyOperationIndices.push_back(
                  payloadIndex->second);
          });
        }
      }

      auto reducerOutputs = linalg.getRegionOutputArgs();
      for (unsigned outputIndex = 0;
           outputIndex < computation.dpsInitOperandNumbers.size();
           ++outputIndex) {
        LinalgReductionOutputFact outputFact;
        outputFact.outputIndex = outputIndex;

        if (reduction.reducerStructureComplete &&
            outputIndex < reducerOutputs.size()) {
          SmallVector<Operation *> combinerOps;
          if (matchReduction(reducerOutputs, outputIndex, combinerOps) &&
              combinerOps.size() == 1) {
            Operation *combiner = combinerOps.front();
            auto combinerIndex = facts.operationIndices.find(combiner);
            if (combinerIndex != facts.operationIndices.end()) {
              outputFact.combinerOperationIndex = combinerIndex->second;
              std::optional<TypedAttr> neutral =
                  getScalarNeutralElement(combiner);
              if (neutral) {
                outputFact.neutralElement = *neutral;
                if (outputIndex < reduction.initOperandValueIndices.size()) {
                  size_t initValueIndex =
                      reduction.initOperandValueIndices[outputIndex];
                  if (initValueIndex < facts.values.size()) {
                    auto seed = getTensorFillConstant(
                        facts.values[initValueIndex].value);
                    if (seed && seed->getType() == neutral->getType())
                      outputFact.seedIdentity =
                          *seed == *neutral
                              ? LinalgReductionSeedIdentity::MatchesNeutral
                              : LinalgReductionSeedIdentity::DiffersFromNeutral;
                  }
                }
              }
            }
          }
        }
        reduction.outputs.push_back(std::move(outputFact));
      }
      computation.reduction = std::move(reduction);
    }
    computation.pureTensor = linalg.hasPureTensorSemantics();
    computation.pureBuffer = linalg.hasPureBufferSemantics();
    if (operation->getNumRegions()) {
      computation.payloadRegion = &operation->getRegion(0);
      computation.payloadMemoryEffectFree = true;
      computation.payloadRegion->walk<WalkOrder::PreOrder>(
          [&](Operation *payloadOperation) {
            auto payloadIndex = facts.operationIndices.find(payloadOperation);
            if (payloadIndex != facts.operationIndices.end())
              computation.payloadOperationIndices.push_back(
                  payloadIndex->second);
            if (!isMemoryEffectFree(payloadOperation))
              computation.payloadMemoryEffectFree = false;
          });
    }
    fact.linalg = std::move(computation);
  }
  return facts;
}

} // namespace buddy::boscame
