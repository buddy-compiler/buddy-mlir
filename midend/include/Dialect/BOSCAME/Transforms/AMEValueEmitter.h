//===- AMEValueEmitter.h - SSA value-semantics AME building blocks --------===//
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
// Shared building blocks that emit BOSC AME operations in the *value*
// semantics introduced by upstream/main (#829): matrix tiles are SSA values of
// type `vector<4x4xT>`, loads return them, MMA consumes and returns them, and
// stores consume them.  The instruction-level register indices
// (`I64Attr:$md`, `I64Attr:$ms1`, lane numbers) no longer exist.
//
// Both FPGA-aware passes (LowerLinalgToBOSCAME and LowerQwenW8A8ToBOSCAME) use
// this layer so that:
//
//   * exactly one place chooses the `mtype` configuration convention,
//   * exactly one place knows the accumulator is an i32 SSA chain while the
//     memory side is fp32,
//   * every failure path can report why a shape or type combination was
//     rejected instead of silently emitting a wrong encoding.
//
// These helpers do NOT create loops.  Wiring an accumulator across an
// `scf.for` reduction is the caller's job (`iter_args` / `scf.yield`), because
// only the caller knows how many accumulator chains a schedule has.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_DIALECT_BOSCAME_AMEVALUEEMITTER_H
#define BUDDY_DIALECT_BOSCAME_AMEVALUEEMITTER_H

#include "Dialect/BOSCAME/Transforms/FPGAAMETarget.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace buddy {
namespace boscame {

using mlir::FailureOr;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::Type;
using mlir::Value;
using mlir::VectorType;

namespace ame {

/// Matrix tile type of the value-semantics layer.
///
/// The current layer represents every tile as `vector<4x4xT>`, which is the
/// *register* granularity; the active tile shape (16x16, 16x64, ...) lives in
/// `bosc_ame.msettile*` plus the target contract, not in the type.  Keeping
/// this in one helper means a future rectangular-tile ABI only has to change
/// here (see Q2 in the migration plan).
VectorType getMatrixTileType(MLIRContext *context, Type elementType);

/// Byte stride of `memref` along `dim`, as an i64 SSA value.
///
/// Uses the strided metadata descriptor rather than assuming a row-major
/// layout, so padded leading dimensions, subviews and reinterpret_cast are all
/// handled.  `dim` is the *memref* dimension, not the logical matrix axis: a
/// physically transposed B passes the dimension that is contiguous.
Value createByteStride(OpBuilder &builder, Location loc, Value memref,
                       unsigned dim = 0);

/// `mlae*.m`: load the left matrix tile.
FailureOr<Value> createLoadA(OpBuilder &builder, Location loc, Type elementType,
                             Value source, Value byteStride,
                             Operation *anchor = nullptr);

/// `mlbe*.m`: load the right matrix tile from a row-major `[K, N]` view.
FailureOr<Value> createLoadB(OpBuilder &builder, Location loc, Type elementType,
                             Value source, Value byteStride,
                             Operation *anchor = nullptr);

/// `mlbte*.m`: load the right matrix tile from a physically transposed
/// `[N, K]` view (the Triton weight layout).
FailureOr<Value> createLoadBTransposed(OpBuilder &builder, Location loc,
                                       Type elementType, Value source,
                                       Value byteStride,
                                       Operation *anchor = nullptr);

/// `mlce*.m`: load an accumulator tile.
///
/// `accElementType` is the accumulator datapath width (i32 on the FPGA), while
/// `memoryElementType` is what the memref actually holds.  On the FPGA these
/// differ on purpose: `mlce32.m` reads an fp32 buffer and uses its bit pattern
/// as the integer accumulator, so only a `+0.0` buffer is a valid zero seed.
FailureOr<Value> createLoadAccumulator(OpBuilder &builder, Location loc,
                                       Type accElementType,
                                       Type memoryElementType, Value source,
                                       Value byteStride,
                                       Operation *anchor = nullptr);

/// MMA: returns the updated accumulator, which must feed the next MMA of the
/// same chain or the final store.  Dropping the result silently loses the
/// accumulation, so callers must thread it explicitly.
FailureOr<Value> createMma(OpBuilder &builder, Location loc, Value acc,
                           Value lhs, Value rhs, Operation *anchor = nullptr);

/// `msce*.m`: store an accumulator tile.  The memory side may be a different
/// (wider/narrower) element type than the accumulator datapath; see
/// `createLoadAccumulator`.
LogicalResult createStoreAccumulator(OpBuilder &builder, Location loc,
                                     Value src, Value dest, Value byteStride,
                                     Operation *anchor = nullptr);

/// Program `mtilem` / `mtilen` / `mtilek`.  These are plain CSR writes and are
/// identical for both profiles; they are wrapped here so every emitter uses
/// the same i64 casting.
LogicalResult configureTiles(OpBuilder &builder, Location loc, Value m,
                             Value n, Value k = {});

/// Program only `mtilek` (the K loop re-sets the K extent for each tile).
LogicalResult configureTileK(OpBuilder &builder, Location loc, Value k);

/// Program the `mtype` CSR for the MMA datapath (the *input* element type).
///
///  * `AmeTargetProfile::Upstream` -> `bosc_ame.msettypei` with the raw element
///    width from `getMsetTypeImm`, exactly as upstream/main emits it.
///  * `AmeTargetProfile::Qwen3Fpga` -> `bosc_ame.msettype` with the bit-field
///    CSR value from `FpgaMtype`.
FailureOr<int64_t> configureMmaType(OpBuilder &builder, Location loc,
                                    Type elementType, AmeTargetProfile profile,
                                    Operation *anchor = nullptr);

/// Program the `mtype` CSR for the accumulator datapath (the *accumulator*
/// element type).  Same split as `configureMmaType`.
FailureOr<int64_t> configureAccumulatorType(OpBuilder &builder, Location loc,
                                            Type accElementType,
                                            AmeTargetProfile profile,
                                            Operation *anchor = nullptr);

/// Raw element width used by the upstream `bosc_ame.msettypei` pathway.
FailureOr<int64_t> getUpstreamMsetTypeImm(Type elementType);

} // namespace ame
} // namespace boscame
} // namespace buddy

#endif // BUDDY_DIALECT_BOSCAME_AMEVALUEEMITTER_H
