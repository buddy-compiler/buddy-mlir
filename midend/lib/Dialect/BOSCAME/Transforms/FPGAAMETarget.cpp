//===- FPGAAMETarget.cpp - AME target profile and mtype encoding ----------===//
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

#include "Dialect/BOSCAME/Transforms/FPGAAMETarget.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"

using namespace mlir;

namespace buddy {
namespace boscame {

llvm::StringRef stringifyAmeTargetProfile(AmeTargetProfile profile) {
  switch (profile) {
  case AmeTargetProfile::Upstream:
    return "upstream";
  case AmeTargetProfile::Qwen3Fpga:
    return "qwen3-fpga";
  }
  llvm_unreachable("unknown AmeTargetProfile");
}

std::optional<AmeTargetProfile> symbolizeAmeTargetProfile(llvm::StringRef name) {
  if (name.empty() || name == "upstream")
    return AmeTargetProfile::Upstream;
  if (name == "qwen3-fpga")
    return AmeTargetProfile::Qwen3Fpga;
  return std::nullopt;
}

FailureOr<AmeTargetProfile> resolveAmeTarget(Operation *op,
                                             llvm::StringRef option) {
  std::optional<AmeTargetProfile> fromOption;
  if (!option.empty()) {
    fromOption = symbolizeAmeTargetProfile(option);
    if (!fromOption)
      return op->emitError()
             << "unknown " << kAmeTargetAttrName << " value '" << option
             << "' (expected 'upstream' or 'qwen3-fpga')";
  }

  std::optional<AmeTargetProfile> fromAttribute;
  if (auto attr = op->getAttrOfType<StringAttr>(kAmeTargetAttrName)) {
    fromAttribute = symbolizeAmeTargetProfile(attr.getValue());
    if (!fromAttribute)
      return op->emitError()
             << "unknown " << kAmeTargetAttrName << " value '"
             << attr.getValue() << "' (expected 'upstream' or 'qwen3-fpga')";
  }

  if (fromOption && fromAttribute && *fromOption != *fromAttribute)
    return op->emitError()
           << "conflicting AME target: pass option requests '"
           << stringifyAmeTargetProfile(*fromOption) << "' but "
           << kAmeTargetAttrName << " is '"
           << stringifyAmeTargetProfile(*fromAttribute) << "'";

  if (fromOption)
    return *fromOption;
  if (fromAttribute)
    return *fromAttribute;
  // Default: upstream/main behaviour.  FPGA semantics are opt-in so that the
  // default pipeline keeps producing GEM5-compatible output.
  return AmeTargetProfile::Upstream;
}

FailureOr<int64_t> getFpgaMtypeImm(Type elementType, FpgaMtypePhase phase) {
  switch (phase) {
  case FpgaMtypePhase::Mma:
    if (elementType.isInteger(8))
      return FpgaMtype::mmaI8;
    break;
  case FpgaMtypePhase::Accumulator:
    if (elementType.isInteger(32))
      return FpgaMtype::accumulatorI32;
    break;
  }
  return failure();
}

bool isFpgaMmaSupported(Type lhsElementType, Type accElementType) {
  // Verified FPGA datapath: i8 x i8 -> i32 (`mqma.b.mm`).
  return lhsElementType.isInteger(8) && accElementType.isInteger(32);
}

bool isFpgaAccumulatorMemorySupported(Type accElementType,
                                      Type memoryElementType) {
  // The FPGA `mlce32.m` / `msce32.m` pair is a 32-bit accumulator datapath
  // whose memory side is fp32: the load interprets the fp32 bit pattern, the
  // store converts i32 -> f32.  Zero initialisation therefore has to be a
  // +0.0 buffer, which is exactly the i32 zero accumulator.
  if (!accElementType.isInteger(32))
    return false;
  return memoryElementType.isF32() || memoryElementType.isInteger(32);
}

} // namespace boscame
} // namespace buddy
