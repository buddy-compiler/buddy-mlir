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

#include <string>

using namespace mlir;

namespace buddy {
namespace boscame {

llvm::StringRef stringifyAmeTargetProfile(AmeTargetProfile profile) {
  switch (profile) {
  case AmeTargetProfile::Upstream:
    return "upstream";
  case AmeTargetProfile::Qwen3Fpga:
    return "qwen3-fpga";
  case AmeTargetProfile::NrFpga:
    return "nr-fpga";
  }
  llvm_unreachable("unknown AmeTargetProfile");
}

std::optional<AmeTargetProfile>
symbolizeAmeTargetProfile(llvm::StringRef name) {
  if (name.empty() || name == "upstream")
    return AmeTargetProfile::Upstream;
  if (name == "qwen3-fpga")
    return AmeTargetProfile::Qwen3Fpga;
  if (name == "nr-fpga")
    return AmeTargetProfile::NrFpga;
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
             << "' (expected 'upstream', 'qwen3-fpga' or 'nr-fpga')";
  }

  std::optional<AmeTargetProfile> fromAttribute;
  if (op->hasAttr(kAmeTargetAttrName) &&
      !op->getAttrOfType<StringAttr>(kAmeTargetAttrName))
    return op->emitError() << kAmeTargetAttrName << " must be a string";
  if (auto attr = op->getAttrOfType<StringAttr>(kAmeTargetAttrName)) {
    fromAttribute = symbolizeAmeTargetProfile(attr.getValue());
    if (!fromAttribute)
      return op->emitError()
             << "unknown " << kAmeTargetAttrName << " value '"
             << attr.getValue()
             << "' (expected 'upstream', 'qwen3-fpga' or 'nr-fpga')";
  }

  if (fromOption && fromAttribute && *fromOption != *fromAttribute)
    return op->emitError() << "conflicting AME target: pass option requests '"
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

/// Element type that the FPGA prototype convention accepts for one operand or
/// result role of an AME matrix operation.
enum class AmeRole { Tile, Accumulator };

/// Roles of the matrix operands and results of one AME operation.  An MMA has
/// both an accumulator operand and tile operands, so the role is tracked per
/// operand rather than per operation.
struct AmeRoleSpec {
  std::optional<AmeRole> operand[3];
  std::optional<AmeRole> result;
};

/// Classify an AME operation by mnemonic.  Returns std::nullopt for operations
/// the FPGA convention does not support.
static std::optional<AmeRoleSpec> classifyAmeMnemonic(llvm::StringRef name) {
  AmeRoleSpec spec;

  // Only the instruction subset emitted by the verified W8A8 adapter has an
  // FPGA intrinsic and fixed-slot contract. Other upstream operations remain
  // available under the upstream profile.
  if (name == "bosc_ame.mlae8.m" || name == "bosc_ame.mlbe8.m" ||
      name == "bosc_ame.mlbte8.m") {
    spec.result = AmeRole::Tile;
    return spec;
  }
  if (name == "bosc_ame.mlce32.m") {
    spec.result = AmeRole::Accumulator;
    return spec;
  }
  if (name == "bosc_ame.msce32.m") {
    spec.operand[0] = AmeRole::Accumulator;
    return spec;
  }
  if (name == "bosc_ame.mqma.b.mm") {
    spec.operand[0] = AmeRole::Accumulator;
    spec.operand[1] = AmeRole::Tile;
    spec.operand[2] = AmeRole::Tile;
    spec.result = AmeRole::Accumulator;
    return spec;
  }

  return std::nullopt;
}

static bool roleMatches(AmeRole role, Type elementType) {
  return role == AmeRole::Tile ? elementType.isInteger(8)
                               : elementType.isInteger(32);
}

LogicalResult verifyFpgaAmeCapabilities(Operation *root) {
  FailureOr<AmeTargetProfile> profile = resolveAmeTarget(root, "");
  if (failed(profile))
    return failure();
  bool nr = *profile == AmeTargetProfile::NrFpga;
  WalkResult result = root->walk([&](Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (!name.starts_with("bosc_ame."))
      return WalkResult::advance();

    // The default profile deliberately leaves no module marker. Nevertheless,
    // already-lowered upstream configuration must never be reinterpreted as
    // FPGA code by a subsequent pass (including after intrinsic export).
    if (name == "bosc_ame.msettypei" || name == "bosc_ame.intr.msettypei" ||
        name == "bosc_ame.msettypehi" || name == "bosc_ame.intr.msettypehi") {
      op->emitError() << "upstream AME configuration cannot be mixed with the "
                      << stringifyAmeTargetProfile(*profile) << " target";
      return WalkResult::interrupt();
    }

    if (nr && name.contains("mlbte")) {
      op->emitError("nr-fpga does not support transposed B loads; use a "
                    "physical [N, K] weight or pack the B tile first");
      return WalkResult::interrupt();
    }

    if (nr && (name.ends_with("msettilemi") ||
               name.ends_with("msettileni") ||
               name.ends_with("msettileki"))) {
      op->emitError("nr-fpga requires register-form tile configuration");
      return WalkResult::interrupt();
    }

    // Only operations that carry a matrix value are constrained by the register
    // file convention.  Configuration instructions and the high-level W8A8
    // semantic ops (quantize_per_group, w8a8_linear, ...) work on memrefs and
    // are lowered by their own patterns.
    auto isMatrix = [nr](Type type) {
      auto vectorType = dyn_cast<VectorType>(type);
      return vectorType && (nr || vectorType.getRank() == 2);
    };
    bool carriesMatrix = llvm::any_of(op->getOperandTypes(), isMatrix) ||
                         llvm::any_of(op->getResultTypes(), isMatrix);
    if (!carriesMatrix)
      return WalkResult::advance();

    // Exported matrix values are rank-one scalable vectors. Check them too:
    // accepting an already-exported upstream load would select the wrong
    // register bank even though the module was subsequently labelled NR.
    bool intrinsic = nr && name.starts_with("bosc_ame.intr.");
    std::string highLevelName;
    if (intrinsic) {
      if (!name.starts_with("bosc_ame.intr.fpga.")) {
        op->emitError("nr-fpga cannot use an upstream matrix intrinsic");
        return WalkResult::interrupt();
      }
      highLevelName =
          "bosc_ame." +
          name.drop_front(StringRef("bosc_ame.intr.fpga.").size()).str();
      name = highLevelName;
    }

    std::optional<AmeRoleSpec> spec = classifyAmeMnemonic(name);
    if (!spec) {
      op->emitError() << "BOSC AME operation '" << name
                      << "' is not supported by the FPGA prototype convention "
                         "(i8 A/B tiles with an i32 accumulator); see "
                         "docs/BOSCAMEFPGAValueSemantics.md";
      return WalkResult::interrupt();
    }

    auto check = [&](Value value, std::optional<AmeRole> role) -> bool {
      if (!role)
        return true;
      auto vectorType = dyn_cast<VectorType>(value.getType());
      if (!vectorType)
        return true;
      if (roleMatches(*role, vectorType.getElementType()))
        return true;
      // The streamed type already carries its own quotes.
      op->emitError() << "BOSC AME operation '" << name << "' uses "
                      << vectorType.getElementType()
                      << " where the FPGA prototype convention requires "
                      << (*role == AmeRole::Tile ? "an 8-bit tile"
                                                 : "a 32-bit accumulator");
      return false;
    };

    for (unsigned index = 0; index < op->getNumOperands() && index < 3; ++index)
      if (!check(op->getOperand(index), spec->operand[index]))
        return WalkResult::interrupt();
    for (Value resultValue : op->getResults())
      if (!check(resultValue, spec->result))
        return WalkResult::interrupt();

    if (nr && !intrinsic &&
        (name == "bosc_ame.mlae8.m" || name == "bosc_ame.mlbe8.m") &&
        !cast<ShapedType>(op->getOperand(0).getType())
             .getElementType().isInteger(8)) {
      op->emitError("nr-fpga A/B tile memory must be i8");
      return WalkResult::interrupt();
    }
    if (!intrinsic &&
        (name == "bosc_ame.mlce32.m" || name == "bosc_ame.msce32.m")) {
      unsigned memoryIndex = name == "bosc_ame.mlce32.m" ? 0 : 1;
      auto memoryType = cast<ShapedType>(op->getOperand(memoryIndex).getType());
      if (nr && !memoryType.getElementType().isInteger(32)) {
        op->emitError("nr-fpga accumulator memory must be i32; msce32.m "
                      "stores raw integer bits, not f32 values");
        return WalkResult::interrupt();
      }
      if (!nr && !memoryType.getElementType().isF32()) {
        op->emitError(
            "qwen3-fpga accumulator memory must be f32; "
            "msce32.m converts i32 to f32 and is not a lossless spill");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
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
  return memoryElementType.isF32();
}

} // namespace boscame
} // namespace buddy
