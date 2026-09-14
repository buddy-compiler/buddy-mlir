//===- FPGAAMETarget.h - AME target profile and mtype encoding --*- C++ -*-===//
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
// Single source of truth for "which AME hardware contract does this module
// target".
//
// buddy-mlir carries two AME software conventions that cannot be mixed inside
// one module:
//
//  * `upstream`  - the value-semantics pathway merged from upstream/main
//                  (#829).  Configuration is the raw element width
//                  (`bosc_ame.msettypei 8`), tile shapes are expressed through
//                  `bosc_ame.msettile*`, and the LLVM backend owns the mapping
//                  from the SSA matrix value to a physical register.
//
//  * `qwen3-fpga` - the Qwen3 FPGA RTL contract.  Configuration is the
//                  bit-field `mtype` CSR written from a register
//                  (`bosc_ame.msettype`), i8 x i8 MMA accumulates into an i32
//                  accumulator that stays resident across the whole K
//                  reduction, and the final `msce32.m` write-back performs the
//                  i32 -> f32 conversion in hardware.
//
// Feeding the wrong convention to the wrong machine is *silent*: both
// encodings are legal i64 constants, so a mismatch produces wrong numbers
// instead of a compile error.  That is why the profile is resolved once, from
// a module attribute, validated against the pass option, and then threaded
// through every emitter instead of being hard-coded per call site.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_DIALECT_BOSCAME_FPGAAMETARGET_H
#define BUDDY_DIALECT_BOSCAME_FPGAAMETARGET_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace buddy {
namespace boscame {

using mlir::FailureOr;
using mlir::Operation;
using mlir::Type;

/// Module attribute that selects the AME hardware contract.
///
///   module attributes {bosc_ame.target = "upstream"}     // default
///   module attributes {bosc_ame.target = "qwen3-fpga"}
inline constexpr llvm::StringLiteral kAmeTargetAttrName = "bosc_ame.target";

enum class AmeTargetProfile {
  /// Default: upstream/main value semantics, GEM5-compatible.
  Upstream,
  /// Qwen3 FPGA RTL: bit-field `mtype` CSR and the FPGA W8A8 schedule.
  Qwen3Fpga,
};

llvm::StringRef stringifyAmeTargetProfile(AmeTargetProfile profile);
std::optional<AmeTargetProfile> symbolizeAmeTargetProfile(llvm::StringRef name);

/// Resolve the AME profile of `op` (usually the module being lowered).
///
/// `option` is the raw pass-option value; an empty string means "not set".
/// Resolution rules:
///   * option set, attribute set, values differ  -> failure (diagnostic)
///   * option set                                -> option wins
///   * attribute set                             -> attribute wins
///   * neither                                   -> Upstream
/// An unparsable value on either side is a failure.
FailureOr<AmeTargetProfile> resolveAmeTarget(Operation *op,
                                             llvm::StringRef option);

/// Phase of the FPGA datapath that the `mtype` CSR is being programmed for.
enum class FpgaMtypePhase {
  /// i8 x i8 -> i32 MMA datapath.
  Mma,
  /// i32 accumulator datapath (initial load and final write-back).
  Accumulator,
};

/// Bit-field `mtype` CSR encoding for the Qwen3 FPGA RTL.
///
/// Layout (RISC-V Matrix Extension v0.5 / Qwen3 RTL, see
/// `kernel/src/backends/ame/core/ame_core.c`):
///
///   bit 16    : mma  (matrix multiply-accumulate enable)
///   bit 12    : mf64, bit 11: mf32, bit 10: mbf16, bit 9: mf16
///   bit  8    : mint4
///   bit  7    : mint64, bit 6: mint32, bit 5: mint16, bit 4: mint8
///   bits 1:0  : msew (element width: 0=e8, 1=e16, 2=e32, 3=e64)
///
/// The FPGA RTL expects this bit-field value, *not* the raw element width that
/// `getMsetTypeImm()` produces for the upstream pathway.
/// Encode one `mtype` CSR value.
constexpr int64_t encodeFpgaMtype(unsigned msew, unsigned typeBit) {
  return (int64_t{1} << 16) | (int64_t{1} << typeBit) |
         static_cast<int64_t>(msew);
}

struct FpgaMtype {
  /// i8 x i8 MMA datapath: mma=1, mint8=1, msew=0 (0x10010, 65552).
  static constexpr int64_t mmaI8 =
      encodeFpgaMtype(/*msew=*/0, /*typeBit=*/4);
  /// i32 accumulator datapath: mma=1, mint32=1, msew=2 (0x10042, 65602).
  static constexpr int64_t accumulatorI32 =
      encodeFpgaMtype(/*msew=*/2, /*typeBit=*/6);
};

/// `mtype` value for an FPGA phase, or failure with a diagnostic when the
/// element type has no verified FPGA encoding.
FailureOr<int64_t> getFpgaMtypeImm(Type elementType, FpgaMtypePhase phase);

/// True when the FPGA pathway has a verified instruction for an
/// `lhs x lhs -> acc` MMA with these element types.
bool isFpgaMmaSupported(Type lhsElementType, Type accElementType);

/// True when the FPGA pathway has a verified accumulator load/store pair that
/// reads/writes `memoryElementType` for an `accElementType` accumulator.
bool isFpgaAccumulatorMemorySupported(Type accElementType,
                                      Type memoryElementType);

} // namespace boscame
} // namespace buddy

#endif // BUDDY_DIALECT_BOSCAME_FPGAAMETARGET_H
