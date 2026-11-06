//===- BOSCAMEPlanIR.h - Verified planned IR emission ---------------------===//
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
// Verified planned-IR surface: check the complete concrete plan before any
// emission mutates the module, prove exclusive SPM window sharing, and expose
// pass factories / dump helpers for the contract -> plan -> emit pipeline.
// Fixed SPM buffers are passed out of band; neutral interchange IR never
// embeds fixed target windows.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEPLANIR_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEPLANIR_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <memory>

namespace mlir {
class Pass;
}
namespace llvm {
class raw_ostream;
}

namespace buddy::boscame {

// Verify the complete concrete plan before any emission mutates the module.
mlir::LogicalResult verifyBOSCAMEPlan(mlir::ModuleOp module);
// Prove that both private roots and all aliases remain in distinct exclusive
// arms and have only supported synchronous, non-escaping memory uses.
bool canShareExclusiveSPMWindow(mlir::Value first, mlir::Value second);
// Target input constraints are passed out of band: the neutral interchange
// IR never contains fixed target windows. Roots are standard private memrefs
// produced by the compatibility adapter after validating the physical input.
struct FixedSPMBuffer {
  mlir::Value root;
  int64_t bank;
  int64_t offset;
  int64_t capacity;
};

// Physical placement debugging does not affect semantic fusion. Disabling
// automatic placement disables optional SPM reuse optimization; required
// materializations still use DDR, and explicit fixed windows remain valid.
struct PlanOptions {
  bool automaticPlacement = true;
  // V1 is a frozen compatibility/reference planner; production defaults to V2.
  int64_t version = 2;
};

// Select and record target routes on a target-independent buffer contract.
mlir::LogicalResult
planBOSCAME(mlir::ModuleOp module,
            llvm::ArrayRef<FixedSPMBuffer> fixedBuffers = {},
            PlanOptions options = {});
// Report recorded decisions without rerunning analysis or selecting routes.
void printBOSCAMEPlan(mlir::ModuleOp module, llvm::raw_ostream &stream);
// Mechanically expand a verified plan, preserving task/source provenance.
mlir::LogicalResult emitBOSCAMEPlan(mlir::ModuleOp module);
// Pass that runs planning (and optional report) on a ModuleOp.
std::unique_ptr<mlir::Pass> createBOSCAMEPlanPass();
// Pass that expands a previously verified plan.
std::unique_ptr<mlir::Pass> createBOSCAMEEmitPass();
// Pass that verifies recorded plan attributes without emitting.
std::unique_ptr<mlir::Pass> createBOSCAMEPlanVerifyPass();
// Register emit + verify passes with the global Pass registry.
void registerBOSCAMEPlanEmissionPasses();
// Register the planning pass with the global Pass registry.
void registerBOSCAMEPlanPasses();

} // namespace buddy::boscame

#endif
