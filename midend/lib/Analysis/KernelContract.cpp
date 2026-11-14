//===- KernelContract.cpp - Target-independent buffer contract passes -----===//
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
// Public entry for the neutral Kernel Contract used between frontends and the
// BOSCAME planner (see docs/KernelContract.md).
//
// This file only wires MLIR passes and the public API onto KernelContractCore:
//   * --kernel-contract-annotate  -> annotateKernelContract / rebuildPlanned
//   * --kernel-contract-verify    -> verifyKernelContract (+ optional planned)
//
// SourcePolicy treats planner-inserted staging copies
// (contract.provenance.generated_by = "planner-staging") as synthetic
// transfers so they cannot invent a missing source-language boundary event.
// Annotate and verify must share that policy. Logical memref memory spaces
// stay zero in both neutral and planned modes; plan.* attributes are accepted
// only when planned=true and never relax content-version checks.
//
//===----------------------------------------------------------------------===//

#include "Analysis/KernelContract.h"
#include "Analysis/detail/KernelContractCore.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
namespace buddy::contract {
namespace {

// True when the op is a Buddy planner staging copy, not a source boundary.
bool isPlannerStaging(Operation *op) {
  auto provenance = op->getAttrOfType<DictionaryAttr>("contract.provenance");
  auto producer =
      provenance ? provenance.getAs<StringAttr>("generated_by") : StringAttr();
  return producer && producer.getValue() == "planner-staging";
}

detail::SourcePolicy sourcePolicy() { return {isPlannerStaging}; }

// Rebuild or complete contract.* attributes on bufferized Linalg/MemRef/SCF.
struct AnnotatePass : PassWrapper<AnnotatePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnnotatePass)
  StringRef getArgument() const final { return "kernel-contract-annotate"; }
  StringRef getDescription() const final { return "Annotate neutral Contract"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    if (failed(annotateKernelContract(getOperation())))
      signalPassFailure();
  }
};
// Check contract.* against actual reads/writes, versions, and control facts.
// --planned allows plan.* alongside the neutral contract without weakening
// memory or content-version rules.
struct VerifyPass : PassWrapper<VerifyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyPass)
  VerifyPass() = default;
  VerifyPass(const VerifyPass &other) : PassWrapper(other) {}
  Option<bool> planned{*this, "planned", llvm::cl::init(false),
                       llvm::cl::desc("Accept Buddy plan attributes")};
  StringRef getArgument() const final { return "kernel-contract-verify"; }
  StringRef getDescription() const final { return "Verify Contract semantics"; }
  void runOnOperation() override {
    if (failed(verifyKernelContract(getOperation(), planned)))
      signalPassFailure();
  }
};
} // namespace

bool hasEntryArgumentABI(func::FuncOp function) {
  return detail::hasEntryArgumentABI(function);
}

// planned=true rebuilds graph metadata after --bosc-ame-plan without repeating
// frontend normalization; otherwise run the full neutral annotation path.
LogicalResult annotateKernelContract(ModuleOp module, StringRef producer,
                                     bool planned) {
  if (planned)
    return detail::rebuildPlannedContract(module, producer, sourcePolicy());
  return detail::annotateKernelContract(module, producer, sourcePolicy());
}

LogicalResult verifyKernelContract(ModuleOp module, bool planned) {
  if (planned)
    return detail::verifyPlannedContract(module, sourcePolicy());
  return detail::verifyKernelContract(module, sourcePolicy());
}
std::unique_ptr<Pass> createKernelContractAnnotatePass() {
  return std::make_unique<AnnotatePass>();
}
std::unique_ptr<Pass> createKernelContractVerifyPass() {
  return std::make_unique<VerifyPass>();
}
void registerKernelContractPasses() {
  PassRegistration<AnnotatePass>();
  PassRegistration<VerifyPass>();
}
} // namespace buddy::contract
