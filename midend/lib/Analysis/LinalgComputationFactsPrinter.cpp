//===- LinalgComputationFactsPrinter.cpp - SSA fact diagnostics -----------===//
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
// Text dump of a LinalgComputationFacts snapshot for lit / debugging.
//
// Output shape (each line prefixed by `indent`):
//   linalg-ssa-facts: scope=explicit-ssa operations=N values=M linalg-ops=K
//   ir-node[i]: op=... parent=... operands=[...] results=[...]
//               optional memory-event={...} and Linalg kind / maps / reduction
//   ssa-value[j]: type=... source=ir-node#...|block-arg... uses=[...]
//
// The printer never invents edges: missing optional fields print as <none>,
// <unknown>, or are omitted. It does not claim execution order, buffer
// contents, reduction associativity, or placement legality.
//
//===----------------------------------------------------------------------===//

#include "Analysis/LinalgComputationFactsPrinter.h"
#include "Analysis/LinalgComputationFacts.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace buddy::boscame;

static llvm::StringRef getKindName(LinalgComputationKind kind) {
  switch (kind) {
  case LinalgComputationKind::Elementwise:
    return "elementwise";
  case LinalgComputationKind::Reduction:
    return "reduction";
  case LinalgComputationKind::Matmul:
    return "matmul";
  case LinalgComputationKind::Fill:
    return "fill";
  case LinalgComputationKind::Other:
    return "other";
  }
  llvm_unreachable("unknown Linalg computation kind");
}

template <typename Range>
static void printIndices(const Range &values, llvm::raw_ostream &out) {
  out << '[';
  llvm::interleaveComma(values, out);
  out << ']';
}

static void printBooleans(llvm::ArrayRef<bool> values, llvm::raw_ostream &out) {
  out << '[';
  llvm::interleaveComma(values, out,
                        [&](bool value) { out << (value ? "true" : "false"); });
  out << ']';
}

static llvm::StringRef
getSeedIdentityName(LinalgReductionSeedIdentity seedIdentity) {
  switch (seedIdentity) {
  case LinalgReductionSeedIdentity::Unknown:
    return "unknown";
  case LinalgReductionSeedIdentity::MatchesNeutral:
    return "matches";
  case LinalgReductionSeedIdentity::DiffersFromNeutral:
    return "differs";
  }
  llvm_unreachable("unknown Linalg reduction seed identity state");
}

static llvm::StringRef getMemoryEventKindName(LinalgMemoryEventKind kind) {
  switch (kind) {
  case LinalgMemoryEventKind::Copy:
    return "copy";
  case LinalgMemoryEventKind::Fill:
    return "fill";
  }
  llvm_unreachable("unknown Linalg memory event kind");
}

static llvm::StringRef getMemoryEventRoleName(LinalgMemoryEventRole role) {
  switch (role) {
  case LinalgMemoryEventRole::Load:
    return "load";
  case LinalgMemoryEventRole::Store:
    return "store";
  case LinalgMemoryEventRole::Padding:
    return "padding";
  case LinalgMemoryEventRole::Logical:
    return "logical";
  case LinalgMemoryEventRole::Unknown:
    return "unknown";
  }
  llvm_unreachable("unknown Linalg memory event role");
}

static llvm::StringRef
getMemoryEventEffectName(LinalgMemoryEventEffect effect) {
  switch (effect) {
  case LinalgMemoryEventEffect::Read:
    return "read";
  case LinalgMemoryEventEffect::Write:
    return "write";
  case LinalgMemoryEventEffect::Unknown:
    return "unknown";
  }
  llvm_unreachable("unknown Linalg memory event effect");
}

void buddy::boscame::printLinalgComputationFacts(
    const LinalgComputationFacts &facts, llvm::raw_ostream &out,
    llvm::StringRef indent) {
  out << indent << "linalg-ssa-facts: scope=explicit-ssa operations="
      << facts.operations.size() << " values=" << facts.values.size()
      << " linalg-ops="
      << llvm::count_if(facts.operations,
                        [](const auto &op) { return op.linalg.has_value(); })
      << '\n';
  for (auto [index, op] : llvm::enumerate(facts.operations)) {
    out << indent << "ir-node[" << index
        << "]: op=" << op.operation->getName().getStringRef() << " parent=";
    if (op.parentOperationIndex)
      out << *op.parentOperationIndex;
    else
      out << "<none>";
    out << " function=";
    if (op.functionOperationIndex)
      out << *op.functionOperationIndex;
    else
      out << "<none>";
    out << " operands=";
    printIndices(op.operandValueIndices, out);
    out << " results=";
    printIndices(op.resultValueIndices, out);
    if (op.computeId)
      out << " compute-id=" << *op.computeId;
    if (op.originId)
      out << " origin-id=" << *op.originId;
    if (op.memoryEvent) {
      const LinalgMemoryEventFact &event = *op.memoryEvent;
      out << " memory-event={kind=" << getMemoryEventKindName(event.kind)
          << " role=" << getMemoryEventRoleName(event.role)
          << " effect=" << getMemoryEventEffectName(event.effect)
          << " origin-id=";
      if (event.originId)
        out << *event.originId;
      else
        out << "<unknown>";
      out << " inputs=";
      printIndices(event.inputValueIndices, out);
      out << " outputs=";
      printIndices(event.outputValueIndices, out);
      out << '}';
    }
    if (op.linalg) {
      const LinalgComputationFact &linalg = *op.linalg;
      out << " kind=" << getKindName(linalg.kind) << " dps-input-operands=";
      printIndices(linalg.dpsInputOperandNumbers, out);
      out << " dps-init-operands=";
      printIndices(linalg.dpsInitOperandNumbers, out);
      out << " init-payload-reads=";
      printBooleans(linalg.initPayloadReads, out);
      out << " maps=[";
      llvm::interleaveComma(linalg.indexingMaps, out);
      out << "] iterators=[";
      llvm::interleaveComma(linalg.iteratorTypes, out, [&](auto iterator) {
        out << mlir::utils::stringifyIteratorType(iterator);
      });
      out << "] reduction-dimensions=";
      printIndices(linalg.reductionDimensions, out);
      if (linalg.reduction) {
        const LinalgReductionStructureFact &reduction = *linalg.reduction;
        out << " reduction-structure={complete="
            << (reduction.reducerStructureComplete ? "true" : "false")
            << " input-args=";
        printIndices(reduction.reducerInputArgumentValueIndices, out);
        out << " init-args=";
        printIndices(reduction.reducerInitArgumentValueIndices, out);
        out << " init-values=";
        printIndices(reduction.initOperandValueIndices, out);
        out << " yields=";
        printIndices(reduction.yieldedValueIndices, out);
        out << " reducer-body-ops=";
        printIndices(reduction.reducerBodyOperationIndices, out);
        out << " output-facts=[";
        llvm::interleaveComma(reduction.outputs, out, [&](const auto &output) {
          out << "{output-index=" << output.outputIndex << " combiner-op=";
          if (output.combinerOperationIndex)
            out << *output.combinerOperationIndex;
          else
            out << "<unknown>";
          out << " neutral=";
          if (output.neutralElement)
            output.neutralElement->print(out);
          else
            out << "<unknown>";
          out << " seed-neutral=" << getSeedIdentityName(output.seedIdentity)
              << '}';
        });
        out << "]}";
      }
      out << " payload-ops=";
      printIndices(linalg.payloadOperationIndices, out);
      out << " tensor=" << (linalg.pureTensor ? "true" : "false")
          << " buffer=" << (linalg.pureBuffer ? "true" : "false")
          << " payload-memory-effect-free="
          << (linalg.payloadMemoryEffectFree ? "true" : "false");
    }
    out << '\n';
  }
  for (auto [index, value] : llvm::enumerate(facts.values)) {
    out << indent << "ssa-value[" << index << "]: type=" << value.type;
    if (value.isShaped) {
      out << " shape=";
      if (value.ranked)
        printIndices(value.shape, out);
      else
        out << "<unranked>";
      out << " element-type=" << value.elementType;
    }
    out << " source=";
    if (value.producerOperationIndex && value.resultNumber)
      out << "ir-node#" << *value.producerOperationIndex << ".result#"
          << *value.resultNumber;
    else if (value.blockOwnerOperationIndex && value.regionNumber &&
             value.blockNumber && value.argumentNumber)
      out << "ir-node#" << *value.blockOwnerOperationIndex << ".region#"
          << *value.regionNumber << ".block#" << *value.blockNumber << ".arg#"
          << *value.argumentNumber;
    else
      out << "<unknown>";
    if (value.frontendLogicalValueId)
      out << " logical-value=" << *value.frontendLogicalValueId;
    out << " uses=[";
    llvm::interleaveComma(value.uses, out, [&](const auto &use) {
      out << "ir-node#" << use.operationIndex << ".operand#"
          << use.operandNumber;
    });
    out << "]\n";
  }
}
