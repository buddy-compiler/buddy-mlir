//===- LinalgComputationFacts.h - Explicit Linalg SSA facts ---------------===//
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
// Explicit SSA / Linalg structure for one module. Consumed by BOSCAME helpers
// and diagnostics (see LinalgComputationFactsPrinter). Snapshots are stale
// after any IR rewrite; call analyzeLinalgComputationFacts again.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_ANALYSIS_LINALGCOMPUTATIONFACTS_H
#define BUDDY_ANALYSIS_LINALGCOMPUTATIONFACTS_H

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>
#include <cstdint>
#include <optional>

namespace mlir {
class ModuleOp;
class Operation;
class Region;
} // namespace mlir

namespace buddy::boscame {

// Structural categories, not hardware contracts or rematerialization proofs.
// In particular, an all-parallel generic may perform a layout permutation.
enum class LinalgComputationKind {
  Elementwise,
  Reduction,
  Matmul,
  Fill,
  Other
};

enum class LinalgMemoryEventKind { Copy, Fill };
enum class LinalgMemoryEventRole { Load, Store, Padding, Logical, Unknown };
enum class LinalgMemoryEventEffect { Read, Write, Unknown };

LinalgComputationKind
classifyLinalgComputation(mlir::linalg::LinalgOp operation);

struct LinalgValueUseFact {
  size_t operationIndex = 0;
  unsigned operandNumber = 0;
};

// One actual SSA value. Memref values describe descriptors, not their stored
// contents; aliases and different versions of memory are never merged here.
struct LinalgValueFact {
  mlir::Value value;
  mlir::Type type;
  bool isShaped = false;
  bool ranked = false;
  llvm::SmallVector<int64_t> shape;
  // A scalar's element type is its own type. Unranked shaped types retain
  // their element type while ranked remains false and shape stays empty.
  mlir::Type elementType;
  std::optional<size_t> producerOperationIndex;
  std::optional<unsigned> resultNumber;
  std::optional<size_t> blockOwnerOperationIndex;
  std::optional<unsigned> regionNumber;
  std::optional<unsigned> blockNumber;
  std::optional<unsigned> argumentNumber;
  // Only exact result-attribute arity or an entry-function argument binds an
  // existing logical ID. Bufferized task output IDs are not SSA result IDs.
  std::optional<int64_t> frontendLogicalValueId;
  // One entry for every explicit operand use, including repeated operands.
  llvm::SmallVector<LinalgValueUseFact> uses;
};

// Raw SSA structure of a reduction region. The operand/yield graph describes
// the reducer but does not prove associativity or a neutral initialization.
enum class LinalgReductionSeedIdentity {
  Unknown,
  MatchesNeutral,
  DiffersFromNeutral
};

// Facts for one reduction output. A neutral element is a typed arithmetic
// fact; matching the seed only recognizes a directly-filled tensor init.
// Neither fact proves reassociation or bit-for-bit equivalence.
struct LinalgReductionOutputFact {
  unsigned outputIndex = 0;
  std::optional<size_t> combinerOperationIndex;
  std::optional<mlir::TypedAttr> neutralElement;
  LinalgReductionSeedIdentity seedIdentity =
      LinalgReductionSeedIdentity::Unknown;
};

struct LinalgReductionStructureFact {
  llvm::SmallVector<size_t> reducerInputArgumentValueIndices;
  llvm::SmallVector<size_t> reducerInitArgumentValueIndices;
  llvm::SmallVector<size_t> initOperandValueIndices;
  llvm::SmallVector<size_t> yieldedValueIndices;
  // Reducer body operations, excluding the outer linalg.yield terminator.
  llvm::SmallVector<size_t> reducerBodyOperationIndices;
  llvm::SmallVector<LinalgReductionOutputFact> outputs;
  bool reducerStructureComplete = false;
};

// A source-memory copy or Linalg fill linked to its explicit SSA operands.
// Copy effects describe the external event direction; fill effects describe
// writes to the fill output. These links do not establish alias or contents.
struct LinalgMemoryEventFact {
  LinalgMemoryEventKind kind = LinalgMemoryEventKind::Copy;
  LinalgMemoryEventRole role = LinalgMemoryEventRole::Unknown;
  LinalgMemoryEventEffect effect = LinalgMemoryEventEffect::Unknown;
  std::optional<int64_t> originId;
  llvm::SmallVector<size_t> inputValueIndices;
  llvm::SmallVector<size_t> outputValueIndices;
};

// Typed Linalg structure attached to an operation fact. These fields do not
// certify reduction identities, general purity, execution order or legality.
struct LinalgComputationFact {
  LinalgComputationKind kind = LinalgComputationKind::Other;
  llvm::SmallVector<unsigned> dpsInputOperandNumbers;
  llvm::SmallVector<unsigned> dpsInitOperandNumbers;
  llvm::SmallVector<bool> initPayloadReads;
  llvm::SmallVector<mlir::AffineMap> indexingMaps;
  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes;
  llvm::SmallVector<int64_t> reductionDimensions;
  std::optional<LinalgReductionStructureFact> reduction;
  mlir::Region *payloadRegion = nullptr;
  // All operations in the payload, including nested operations/terminators,
  // in the same preorder as the enclosing operation table.
  llvm::SmallVector<size_t> payloadOperationIndices;
  bool pureTensor = false;
  bool pureBuffer = false;
  // False for a nested memory effect or an operation with unknown effects.
  // This is not a speculation or rematerialization proof.
  bool payloadMemoryEffectFree = false;
};

struct LinalgOperationFact {
  mlir::Operation *operation = nullptr;
  std::optional<size_t> parentOperationIndex;
  // Includes the function operation itself; absent outside a function.
  std::optional<size_t> functionOperationIndex;
  llvm::SmallVector<size_t> operandValueIndices;
  llvm::SmallVector<size_t> resultValueIndices;
  std::optional<LinalgComputationFact> linalg;
  std::optional<LinalgMemoryEventFact> memoryEvent;
  // Associations copied from existing metadata, never inferred task IDs.
  std::optional<int64_t> computeId;
  // Existing origin_id/compute_origin_id/memory.origin_id, only when all
  // present, nonnegative origin IDs agree. Conflicts remain unassociated.
  std::optional<int64_t> originId;
};

// A read-only snapshot of explicit SSA structure in one module. Operation
// indices follow preorder and value indices follow first occurrence while
// visiting block arguments, operands and results. Handles are valid only
// until the referenced IR changes; reanalyze after rewrites.
//
// Region arguments and yields remain distinct values. This does not infer
// control-flow joins, recurrence, memory contents, aliases, execution order or
// complete compute graphs, and must not independently authorize residency.
struct LinalgComputationFacts {
  llvm::SmallVector<LinalgOperationFact, 0> operations;
  llvm::SmallVector<LinalgValueFact> values;
  llvm::DenseMap<mlir::Operation *, size_t> operationIndices;
  llvm::DenseMap<mlir::Value, size_t> valueIndices;
};

LinalgComputationFacts analyzeLinalgComputationFacts(mlir::ModuleOp module);

} // namespace buddy::boscame

#endif // BUDDY_ANALYSIS_LINALGCOMPUTATIONFACTS_H
