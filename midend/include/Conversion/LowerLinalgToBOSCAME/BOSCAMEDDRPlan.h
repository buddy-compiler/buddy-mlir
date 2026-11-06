//===- BOSCAMEDDRPlan.h - Exact CPU recipes for DDR ops --------===//
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
// CPU-side recipes for operations that stay on materialized DDR (no SPM).
// A recipe is derived from the actual operation and never authorizes changing
// its arithmetic. Used when the planner proves an op cannot or should not
// move through the AME/SPM path.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEDDRPLAN_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEDDRPLAN_H

#include "mlir/IR/BuiltinOps.h"

namespace buddy::boscame {

// Build the DDR recipe dictionary for `op`, or failure if unsupported.
mlir::FailureOr<mlir::DictionaryAttr> getDDRRecipe(mlir::Operation *op);

// Re-derive and compare the recorded DDR recipe before emission.
mlir::LogicalResult verifyDDRRecipe(mlir::Operation *op);

// Emit the verified DDR recipe, replacing `op`. Returns the new anchor op.
mlir::Operation *emitDDRRecipe(mlir::Operation *op);

} // namespace buddy::boscame

#endif // BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEDDRPLAN_H
