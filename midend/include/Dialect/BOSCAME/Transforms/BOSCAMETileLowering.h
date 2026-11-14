//===- BOSCAMETileLowering.h - Tile ops -> FPGA intrinsics -----===//
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
// Lowering of `bosc_ame.tile.*` to the FPGA AME instruction set.
//
// The patterns in `BOSCAMETileLowering.cpp` share the LLVMTypeConverter and the
// conversion target of `-lower-bosc-ame`; this header is the population
// interface so that the unified pass and the tile-only pass entry point use
// exactly one implementation.
//
// Everything that the hardware contract pins down lives here or in
// `TileOps.h`, never in a pattern body:
//
//   * the module attribute that selects the hardware contract,
//   * the `mtype` values of the two datapath phases.
//
// Register allocation is deliberately *not* part of the interface: the tile
// operations name no register, and the lowering emits the ordinary
// `bosc_ame.intr.*` operations so that the backend allocates tile and
// accumulator registers itself.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_DIALECT_BOSCAME_TRANSFORMS_BOSCAMETILELOWERING_H
#define BUDDY_DIALECT_BOSCAME_TRANSFORMS_BOSCAMETILELOWERING_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"

namespace buddy {
namespace boscame {

// Module attribute that selects the AME hardware contract.
inline constexpr llvm::StringLiteral kAmeTargetAttrName = "bosc_ame.target";

// Name of the contract implemented by this first version.
inline constexpr llvm::StringLiteral kAmeTargetNrFpga = "nr-fpga";

// Hardware contract of the AME datapath the tile operations are lowered
// against.
enum class TileTargetProfile {
  // NH/RA FPGA, W8A8: register-form configuration, raw i32 accumulator
  // memory, fixed `tr0`/`tr4`/`acc0` slots, no transposed B load.
  NrFpga,
};

// Bit-field `mtype` values of the two datapath phases.
//
// Layout (RISC-V Matrix Extension v0.5 / the FPGA RTL):
//
//   bit 16    : mma
//   bit 12..9 : mf64, mf32, mbf16, mf16
//   bit  8    : mint4
//   bit  7..4 : mint64, mint32, mint16, mint8
//   bits 1:0  : msew (0 = e8, 1 = e16, 2 = e32, 3 = e64)
//
// `encodeFpgaMtype` is the single definition of the encoding; the two phases
// below are the only values the tile lowering uses.
constexpr int64_t encodeFpgaMtype(unsigned msew, unsigned typeBit) {
  return (int64_t{1} << 16) | (int64_t{1} << typeBit) |
         static_cast<int64_t>(msew);
}

// i8 x i8 -> i32 MMA phase: `mma=1`, `mint8=1`, `msew=e8` (0x10010).
inline constexpr int64_t kFpgaMtypeMmaI8 = encodeFpgaMtype(0, 4);
// i32 accumulator phase: `mma=1`, `mint32=1`, `msew=e32` (0x10042).
inline constexpr int64_t kFpgaMtypeAccumulatorI32 = encodeFpgaMtype(2, 6);

// How the tile operations are lowered.
struct TileLoweringOptions {
  TileTargetProfile profile = TileTargetProfile::NrFpga;
};

// True when `module` contains at least one `bosc_ame.tile.*` operation.
bool hasTileOps(mlir::ModuleOp module);

// Resolve the AME contract of `module` from `bosc_ame.target`.
//
// Tile operations lower to the FPGA intrinsic set, so they are only accepted
// when the module *explicitly* selects a contract this implementation
// provides; there is no implicit default that could silently pick the wrong
// hardware semantics.  Emits a diagnostic on `module` and fails otherwise.
mlir::LogicalResult resolveTileTargetProfile(mlir::ModuleOp module,
                                             TileTargetProfile &profile);

// Verify the value-level rules of the tile operations.
//
// The first version has no matrix-value ABI, so a matrix value has to be
// produced by a tile operation in the same block and may only feed a
// `bosc_ame.tile.mma` or a `bosc_ame.tile.store_acc`.  This is a scope rule on
// the values; which register holds them is decided by the backend and is not
// checked, produced or named here.
mlir::LogicalResult verifyBOSCAMETileValues(mlir::ModuleOp module);

// Population interface of the tile lowering.  Adds one conversion pattern per
// `bosc_ame.tile.*` operation, all of which use `converter` for the memref
// descriptor and for the matrix register type.
void populateBOSCAMETileLoweringPatterns(mlir::LLVMTypeConverter &converter,
                                         mlir::RewritePatternSet &patterns,
                                         const TileLoweringOptions &options);

} // namespace boscame
} // namespace buddy

#endif // BUDDY_DIALECT_BOSCAME_TRANSFORMS_BOSCAMETILELOWERING_H
