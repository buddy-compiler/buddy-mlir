//===- BOSCAMEStreamPlan.h - Content-version streaming plans --------------===//
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
// Content-version streaming plans: analyze / verify / emit deterministic
// streaming of private buffers without changing logical computations. Fixed
// hardware input windows are never virtualized; unproved components keep
// ordinary buffer representation.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMESTREAMPLAN_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMESTREAMPLAN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace buddy::boscame {

// Record a deterministic, content-version-based streaming plan without
// changing the logical computations. Fixed hardware input windows are not
// candidates for virtualization. Unsupported or unproved components retain
// their ordinary buffer representation.
mlir::LogicalResult analyzeStreams(mlir::ModuleOp module,
                                   llvm::ArrayRef<mlir::Value> fixedRoots = {});

// Recompute the semantic proof and compare every recorded recipe. This must
// succeed before either buffer materialization or logical IR destruction.
mlir::LogicalResult verifyStreams(mlir::ModuleOp module);

// Expand verified sinks, retaining exact scalar payloads and the original
// ordered 16-lane reductions. Archive logical tasks separately from physical
// execution anchors; one virtual producer may map to several anchors.
// inputVerified is reserved for the combined emitter after its full-module
// verification, which may have expanded other disjoint execution groups. In
// that mode consume only the recorded set, still checking member boundaries
// and every recipe, without rerunning source-stability analysis over already
// emitted physical groups.
mlir::LogicalResult emitStreams(mlir::ModuleOp module,
                                bool inputVerified = false);

// True when `operation` belongs to a verified streaming recipe group.
bool isStreamMember(mlir::Operation *operation);
// True when `value` is a virtual root owned by a streaming recipe.
bool isStreamVirtual(mlir::Value value);
// True when `operation` defines a virtual root owned by a streaming recipe.
bool isStreamVirtual(mlir::Operation *operation);

} // namespace buddy::boscame

#endif
