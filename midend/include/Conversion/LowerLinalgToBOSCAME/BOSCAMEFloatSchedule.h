//===- BOSCAMEFloatSchedule.h - Ordered scalar float microtiles -------===//
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
// Analyze / verify / emit ordered scalar floating-point register microtile
// schedules. Fixed hardware roots in `fixedRoots` are excluded from
// virtualization. Membership helpers let later passes skip ordinary buffer
// materialization for covered ops.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEFLOATSCHEDULE_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEFLOATSCHEDULE_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"

namespace buddy::boscame {

// Prove float microtile schedules and record recipes on the module.
mlir::LogicalResult
analyzeFloatSchedules(mlir::ModuleOp module,
                      llvm::ArrayRef<mlir::Value> fixedRoots = {});

// Recompute proofs and compare every recorded float schedule recipe.
mlir::LogicalResult verifyFloatSchedules(mlir::ModuleOp module);

// Expand verified float schedules into executable IR.
mlir::LogicalResult emitFloatSchedules(mlir::ModuleOp module);

// True when `op` belongs to a verified float schedule group.
bool isFloatScheduleMember(mlir::Operation *op);

// True when `op` is a virtual root owned by a float schedule.
bool isFloatScheduleVirtual(mlir::Operation *op);

} // namespace buddy::boscame

#endif // BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEFLOATSCHEDULE_H
