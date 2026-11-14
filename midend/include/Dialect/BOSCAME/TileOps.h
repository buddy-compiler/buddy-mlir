//===- TileOps.h - SPM <-> AME tile operation contracts --------===//
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
// The `bosc_ame.tile.*` operations are the typed boundary between SPM buffers
// and the AME matrix register file.  This header owns the *contract* of that
// boundary - which memref layouts, element types and tile shapes the operations
// accept, and which byte stride the corresponding hardware load/store needs -
// so that the op verifiers and the LLVM lowering cannot drift apart.
//
// The two rules that are easy to get wrong, and are therefore encoded here
// instead of at each call site:
//
//   * A physical row stride is the byte distance between two *hardware* rows.
//     For A and the accumulator that is the leading stride; for B, whose
//     storage is an `N x K` buffer with K contiguous while the logical matrix
//     is `K x N`, it is the trailing stride.
//   * Every tile operand must be a ranked, statically shaped, 2-D memref.  The
//     default storage space is `#bosc_ame.spm<bank>`; a placement pass may add
//     `bosc_ame.direct_ddr` to use an externally supplied, proven-aligned DDR
//     descriptor directly.  A dynamic shape cannot be compiled into the tile
//     configuration.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_DIALECT_BOSCAME_TILEOPS_H
#define BUDDY_DIALECT_BOSCAME_TILEOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include <array>
#include <cstdint>
#include <optional>

namespace buddy {
namespace boscame {

// Maximum extent of one AME tile dimension (`TMMAX`/`TNMAX`/`TKMAX`).
inline constexpr int64_t kAmeMaxTileDim = 64;

// Role of a tile value in the AME datapath.  The role selects the hardware
// register file, the element type and the physical row stride.
enum class TileRole {
  // Left input tile `A[M, K]`: K contiguous, row stride = leading stride.
  A,
  // Right input tile `B[K, N]`: the storage is `N x K` with K contiguous, so
  // the row stride is the *trailing* stride.
  B,
  // `C`/accumulator tile `[M, N]`: N contiguous, row stride = leading stride.
  Accumulator,
};

llvm::StringRef stringifyTileRole(TileRole role);

// The single element type the first-version AME datapath implements for a
// role, or `std::nullopt` when `elementType` is not it.  The AME instruction
// set has wider variants, but a variant is only accepted by this dialect once
// its intrinsic, its register mapping and its hardware contract are all in
// place; until then the element type selects the instruction and the `mtype`
// phase, so guessing would silently pick the wrong datapath.
std::optional<unsigned> getTileElementByteWidth(TileRole role,
                                                mlir::Type elementType);

// Byte footprint of an SPM tile operand as the hardware sees it.
struct TileStorageLayout {
  // Logical shape, e.g. `[M, K]` for A and the `[K, N]` of the logical B.
  std::array<int64_t, 2> shape = {0, 0};
  // Element strides of the logical memref type.
  std::array<int64_t, 2> strides = {0, 0};
  // Size of one element in bytes.
  int64_t elementBytes = 0;
  // Byte distance between two consecutive *physical* rows.  This is the value
  // the AME load/store instruction takes as its stride operand.
  int64_t rowStrideBytes = 0;
  // Descriptor offset of the memref in bytes.
  int64_t byteOffset = 0;

  int64_t rows() const { return shape[0]; }
  int64_t cols() const { return shape[1]; }
};

// The AME row-stride of a role: the leading stride for A/accumulator, the
// trailing stride for B.  Exposed because it is the one place where the
// physical B layout differs from its logical description.
constexpr int tileRowStrideIndex(TileRole role) {
  return role == TileRole::B ? 1 : 0;
}

// Physical layout of an AME tile operand, or `std::nullopt` when the type is
// not one the AME tile operations accept.  Pure function: the diagnostics for
// a rejected type live in `verifyTileStorageLayout`.
std::optional<TileStorageLayout> computeTileStorageLayout(mlir::MemRefType type,
                                                          TileRole role);

// `computeTileStorageLayout` with diagnostics emitted on `op`.
mlir::LogicalResult verifyTileStorageLayout(mlir::Operation *op,
                                            mlir::MemRefType type,
                                            TileRole role,
                                            TileStorageLayout &layout);

// Verify the memory-side operand of a tile load/store: storage space, rank,
// static shape, strided layout, element type and tile size.
mlir::LogicalResult verifyTileMemRefOperand(mlir::Operation *op,
                                            mlir::Value operand, TileRole role,
                                            TileStorageLayout &layout);

// Verify a matrix value against the role's expected shape and element type.
// `shape` is filled with the logical shape of the value.
mlir::LogicalResult verifyTileMatrixValue(mlir::Operation *op,
                                          mlir::Value value, TileRole role,
                                          std::array<int64_t, 2> &shape);

} // namespace boscame
} // namespace buddy

#endif // BUDDY_DIALECT_BOSCAME_TILEOPS_H
