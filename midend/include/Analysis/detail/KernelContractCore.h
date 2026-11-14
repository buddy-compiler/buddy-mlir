//===- KernelContractCore.h - Buffer Contract implementation --------------===//
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
// Internal implementation of Kernel Contract annotate / verify.
// Analysis/KernelContract.h is the stable public surface; this header is for
// the core library and for callers that must inject a SourcePolicy (planner
// staging copies, tests).
//
// Responsibilities:
//   * Walk bufferized Linalg / memref / scf structure
//   * Attach or check contract.* attributes (tasks, access, values, ABI)
//   * Enforce content-version and entry ABI rules
//
// Annotation and verification must share the same SourcePolicy so synthetic
// transfers cannot invent a missing source-language boundary event.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_ANALYSIS_DETAIL_KERNELCONTRACTCORE_H
#define BUDDY_ANALYSIS_DETAIL_KERNELCONTRACTCORE_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class ModuleOp;
class Operation;
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace buddy::contract::detail {

// Policy hook shared by annotate and verify.
// If `isSyntheticTransfer` returns true for an op, that op is treated as a
// compiler-inserted transfer and cannot stand in for a source-language
// boundary event when rebuilding or checking lineage.
struct SourcePolicy {
  bool (*isSyntheticTransfer)(mlir::Operation *) = nullptr;
};

// Validate contract.version / producer compatibility on the module.
mlir::LogicalResult checkKernelContractVersions(mlir::ModuleOp module);

// True when `function` is an unused entry that may carry module ABI promises.
bool hasEntryArgumentABI(mlir::func::FuncOp function);

// Neutral annotate / verify (logical memory space stays zero).
mlir::LogicalResult annotateKernelContract(mlir::ModuleOp module,
                                           llvm::StringRef producer,
                                           SourcePolicy sourcePolicy = {});
mlir::LogicalResult verifyKernelContract(mlir::ModuleOp module,
                                         SourcePolicy sourcePolicy = {});

// After planning: rebuild graph metadata without repeating producer
// normalization. Existing plan.* attributes are accepted; logical memory
// space must remain zero. Content-version rules are not relaxed.
mlir::LogicalResult rebuildPlannedContract(mlir::ModuleOp module,
                                           llvm::StringRef producer,
                                           SourcePolicy sourcePolicy);
mlir::LogicalResult verifyPlannedContract(mlir::ModuleOp module,
                                          SourcePolicy sourcePolicy);

} // namespace buddy::contract::detail

#endif // BUDDY_ANALYSIS_DETAIL_KERNELCONTRACTCORE_H
