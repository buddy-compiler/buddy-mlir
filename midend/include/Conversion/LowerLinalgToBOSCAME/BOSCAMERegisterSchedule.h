//===- BOSCAMERegisterSchedule.h - Register-resident AME recipes ----------===//
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
// Integer AME register-resident schedules: prove and record recipes without
// changing the logical graph, verify before mutation, then emit physical
// groups (`bosc_ame.execution_id` / `register_tasks`) before buffer
// materialization. Run after accumulator fusion.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEREGISTERSCHEDULE_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEREGISTERSCHEDULE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace buddy::boscame {

// Prove and record integer AME schedules without changing the logical graph.
// Run after accumulator fusion and before snapshot forwarding/placement. A
// second call refreshes recipes after contract task IDs have been rebuilt.
mlir::LogicalResult
analyzeRegisterSchedules(mlir::ModuleOp module,
                         llvm::ArrayRef<mlir::Value> fixedRoots = {});

// These logical operations/allocations belong to a verified physical recipe;
// ordinary storage selection and operation emission must leave them alone.
bool isRegisterScheduleMember(mlir::Operation *operation);
// True when `operation` is a virtual root owned by a register schedule.
bool isRegisterScheduleVirtual(mlir::Operation *operation);

// Reconstruct every proof from the logical graph and compare the complete
// recorded recipe, membership and virtual-root ownership before mutation.
mlir::LogicalResult verifyRegisterSchedules(mlir::ModuleOp module);

// Emit register schedules before materializeBuffers. Each physical group has
// bosc_ame.execution_id and bosc_ame.register_tasks; covered contract.task
// facts move to function bosc_ame.register_logical_tasks and are mapped by
// bosc_ame.register_task_map. No covered contract.task remains on a physical
// anchor. The caller excludes register members from its ordinary task set.
mlir::LogicalResult emitRegisterSchedules(mlir::ModuleOp module);

} // namespace buddy::boscame

#endif
