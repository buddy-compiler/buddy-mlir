//===- BOSCAMEHelpers.h - Shared DMA and AME lowering helpers -------------===//
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
// Shared builders and legality helpers used while lowering bufferized Linalg /
// Triton-style loads/stores into BOSCAME SPM views, DMA, and marked matmul.
// Owns per-function SPM workspace accounting, DDR policy emission, vectorized
// copy/fill, and SPM computation placement checks. Callers in
// LowerLinalgToBOSCAME and dialect transforms should use these helpers instead
// of duplicating bank offsets or DMA wait sequences.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_BOSCAMEHELPERS_H
#define BUDDY_CONVERSION_BOSCAMEHELPERS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <array>

namespace buddy::boscame::lowering {

// One static, non-reusing SPM byte budget per function, shared by accesses and
// matmul. Existing `bosc_ame.spm.view` ops seed the budget after bufferization.
class Workspace {
public:
  // Allocate a contiguous SPM view of `type` near `anchor`.
  // `preferredBank` is -1 for any bank, otherwise 0 or 1.
  mlir::FailureOr<mlir::Value> allocate(mlir::OpBuilder &builder,
                                        mlir::Operation *anchor,
                                        mlir::MemRefType type,
                                        int64_t preferredBank = -1);

private:
  // Bytes already claimed in SPM banks 0 and 1, keyed by containing FuncOp.
  llvm::DenseMap<mlir::Operation *, std::array<int64_t, 2>> used;
};

// Insert module-level DDR policy declarations required by later DMA emission.
mlir::LogicalResult declareDdrPolicies(mlir::ModuleOp module);

// Emit a single DDR policy marker (`write` selects store vs load policy).
void emitDdrPolicy(mlir::OpBuilder &builder, mlir::Location loc, bool write);

// Attribute name for CPU DDR policy markers on ops / modules.
inline constexpr llvm::StringLiteral kCpuDdrPolicy = "bosc_ame.cpu_ddr_policy";

// Emit CPU DDR policies for every external DDR endpoint that needs them.
mlir::LogicalResult emitCpuDdrPolicies(mlir::ModuleOp module);

// Emit `bosc_ame.dma.start` + wait for a contiguous `bytes` transfer.
void emitDmaAndWait(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value src, mlir::Value dst, int64_t bytes);

// Both memrefs are rank one. Each transfer is a complete 64-byte vector;
// this helper never emits a scalar tail, including on SPM.
mlir::LogicalResult copyVectors(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value src, mlir::Value dst,
                                int64_t elementCount);

// Vectorized fill of a rank-one buffer; same 64-byte vector granularity.
mlir::LogicalResult fillVectors(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value buffer, mlir::Value value);

// Resolve or allocate the memref backing a tensor store near `anchor`.
mlir::FailureOr<mlir::Value> getStoreBuffer(mlir::OpBuilder &builder,
                                            mlir::Operation *anchor,
                                            mlir::Value tensor);

// Lower a masked pointer load into SPM-backed buffers via `workspace`.
mlir::FailureOr<mlir::Value> lowerLoad(mlir::OpBuilder &builder,
                                       mlir::Operation *anchor, mlir::Value ptr,
                                       mlir::RankedTensorType type,
                                       llvm::ArrayRef<mlir::OpFoldResult> mask,
                                       mlir::Value other, Workspace &workspace);

// Lower a masked pointer store from `tensor` into DDR/SPM via `workspace`.
mlir::LogicalResult lowerStore(mlir::OpBuilder &builder,
                               mlir::Operation *anchor, mlir::Value ptr,
                               mlir::Value tensor,
                               llvm::ArrayRef<mlir::OpFoldResult> mask,
                               Workspace &workspace);

// Attribute marking a matmul that must lower through the AME tile path.
inline constexpr llvm::StringLiteral kLowerToAme = "bosc_ame.lower_to_ame";

// Lower every matmul tagged with `kLowerToAme` into tile/DMA IR.
mlir::LogicalResult lowerMarkedMatmuls(mlir::ModuleOp module);

// Assign SPM buffers to planned SPM computations without emitting tiles yet.
mlir::LogicalResult planSpmComputationBuffers(mlir::ModuleOp module);
// Check the exact operand placement accepted by SPM computation lowering.
// This is a pure legality query: it does not rewrite the operation. Callers
// may use a detached clone to model resident and operation-local buffers.
mlir::LogicalResult checkSpmComputation(mlir::Operation *operation);
// Positional generic inputs whose private fill buffers can stay in registers.
// Null entries retain ordinary buffer loads. A buffer may have successive
// fills: each reader receives the broadcast from its latest dominating fill.
using SpmBroadcastMap =
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Value>>;
// Replace private, disjoint SPM fill materializations with register
// broadcasts. Readers keep their buffer operands until computation lowering
// consumes the map. Preserve fill task IDs and logical dependencies when
// verifying graphs.
mlir::LogicalResult forwardSpmBroadcasts(
    mlir::ModuleOp module, SpmBroadcastMap &broadcasts,
    llvm::function_ref<mlir::LogicalResult(mlir::ModuleOp)> verifyGraph =
        nullptr);
// When supplied, preserve each computation's graph metadata on its unique
// completion operation and verify the graph after each lowering rewrite.
// Static-prefix selects drop their mask dependencies before dead private
// mask tasks are removed and the explicit graph user tables are refreshed.
mlir::LogicalResult lowerSpmComputations(
    mlir::ModuleOp module,
    llvm::function_ref<mlir::LogicalResult(mlir::ModuleOp)> verifyGraph =
        nullptr);
// Forward private rank-zero scalar storage to SSA, preserving task anchors.
void promotePrivateScalarBuffers(mlir::ModuleOp module);
// Remove trivially dead lowering artifacts without erasing graph tasks.
void eraseDeadSpmComputationArtifacts(mlir::ModuleOp module);

// Final structural checks before leaving the SPM computation lowering stage.
mlir::LogicalResult verifyExit(mlir::ModuleOp module);

} // namespace buddy::boscame::lowering

#endif // BUDDY_CONVERSION_BOSCAMEHELPERS_H
