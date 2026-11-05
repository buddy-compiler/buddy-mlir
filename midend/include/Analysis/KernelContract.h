//===- KernelContract.h - Target-independent buffer contracts -------------===//
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
// Public API for the target-independent Kernel Contract used between frontends
// (for example triton-riscv) and the BOSCAME planner. The contract answers what
// buffers exist, which content versions ops read/write, and entry ABI promises.
// It does not encode bank placement, DMA channels, or tile schedules; those
// belong to plan.* attributes after planning.
//
// Passes:
//   * --kernel-contract-annotate  rebuilds / normalizes contract.* metadata
//   * --kernel-contract-verify    checks metadata against SSA and program order
//
// See docs/KernelContract.md. Implementation lives in detail/KernelContractCore
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_ANALYSIS_KERNELCONTRACT_H
#define BUDDY_ANALYSIS_KERNELCONTRACT_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {
class ModuleOp;
class Pass;
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace buddy::contract {

// Module argument ABI promises apply only to entry functions with no symbol
// uses in the module. Called/address-taken helpers can receive aliasing or
// unaligned views, and require separate call-site proofs.
bool hasEntryArgumentABI(mlir::func::FuncOp function);

// Rebuild the buffer Contract on `module`.
// `producer` is recorded on module-level contract metadata.
// When `planned` is false, run full neutral normalization.
// When `planned` is true, skip producer normalization, keep logical memory
// space zero, and recognize planner staging copies via SourcePolicy so they
// cannot invent a missing source-language boundary event.
mlir::LogicalResult
annotateKernelContract(mlir::ModuleOp module,
                       llvm::StringRef producer = "buddy-neutral",
                       bool planned = false);

// Verify the neutral (or planned) contract against actual operands, payloads
// and program order. `planned` permits target plan.* attributes; it does not
// relax memory or content-version correctness. Diagnostics name the failing op.
mlir::LogicalResult verifyKernelContract(mlir::ModuleOp module,
                                         bool planned = false);

// MLIR pass factories for --kernel-contract-annotate /
// --kernel-contract-verify.
std::unique_ptr<mlir::Pass> createKernelContractAnnotatePass();
std::unique_ptr<mlir::Pass> createKernelContractVerifyPass();

// Register both contract passes with the global PassRegistry.
void registerKernelContractPasses();

} // namespace buddy::contract

#endif // BUDDY_ANALYSIS_KERNELCONTRACT_H
