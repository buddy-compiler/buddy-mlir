//====- IMEDialect.cpp - MLIR IME dialect implementation ------------------===//
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

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "Dialect/IME/IMEDialect.h"
#include "Dialect/IME/IMEOps.h"

using namespace mlir;
using namespace buddy::ime;

#include "IME/IMEDialect.cpp.inc"

#define GET_OP_CLASSES
#include "IME/IME.cpp.inc"

void IMEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "IME/IME.cpp.inc"
      >();
}

namespace {
LogicalResult verifyDot(Operation *op) {
  Type result = op->getResult(0).getType();
  if (result != op->getOperand(0).getType())
    return op->emitOpError("result and accumulator must have the same type");
  for (auto [i, value] : llvm::enumerate(op->getOperands())) {
    if (i == 3 && value.getType().isInteger(64))
      continue; // Dynamic K1 slide.
    auto vector = dyn_cast<VectorType>(value.getType());
    if (!vector || vector.getRank() != 1 || !vector.isScalable())
      return op->emitOpError("requires one-dimensional scalable vector tiles");
  }
  // Basic A100 overloads have a fixed 8x16 integer / 8x8 FP16 layout.
  // Keep the legacy K1 vector overloads available for its sliding operations.
  auto c = cast<VectorType>(result);
  auto a = cast<VectorType>(op->getOperand(1).getType());
  auto b = cast<VectorType>(op->getOperand(2).getType());
  if (c.getShape()[0] == 4) {
    bool fp = op->getName().getStringRef() == "ime.intr.vfmadot";
    bool basic = op->getName().getStringRef() == "ime.intr.vmadot" ||
                 op->getName().getStringRef() == "ime.intr.vmadotu" ||
                 op->getName().getStringRef() == "ime.intr.vmadotsu" ||
                 op->getName().getStringRef() == "ime.intr.vmadotus";
    auto ctx = op->getContext();
    Type input =
        fp ? Type(Float16Type::get(ctx)) : Type(IntegerType::get(ctx, 8));
    Type acc =
        fp ? Type(Float32Type::get(ctx)) : Type(IntegerType::get(ctx, 32));
    if ((!fp && !basic) || c.getElementType() != acc ||
        a != VectorType::get({fp ? 4 : 8}, input, true) || a != b)
      return op->emitOpError(
          "invalid A100 dot tile types or unsupported sliding-window overload");
  } else {
    bool fp = op->getName().getStringRef().contains("vfmadot");
    StringRef name = op->getName().getStringRef();
    bool window = name != "ime.intr.vmadot" && name != "ime.intr.vmadotu" &&
                  name != "ime.intr.vmadotsu" && name != "ime.intr.vmadotus" &&
                  name != "ime.intr.vfmadot";
    unsigned width = a.getElementTypeBitWidth();
    bool valid =
        fp ? c.getShape()[0] == 16 && c.getElementType().isF16() &&
                 a.getElementType().isF16() && b.getElementType().isF16()
           : c.getShape()[0] == 8 && c.getElementType().isSignlessInteger(32) &&
                 (width == 8 || width == 16) &&
                 a.getElementType().isSignlessInteger();
    int64_t lanes = fp ? 16 : 256 / width;
    if (!valid || a.getElementType() != b.getElementType() ||
        a.getShape()[0] != lanes * (window ? 2 : 1) || b.getShape()[0] != lanes)
      return op->emitOpError("unsupported K1 dot vector types");
  }
  return success();
}

LogicalResult verifyA100(Operation *op) {
  auto a = cast<VectorType>(op->getOperand(0).getType());
  auto b = cast<VectorType>(op->getOperand(1).getType());
  auto out = cast<VectorType>(op->getResult(0).getType());
  auto isTile = [](VectorType type) {
    return type.getRank() == 1 && type.isScalable() &&
           type.getElementType().isSignlessInteger();
  };
  if (!isTile(a) || !isTile(b) || !isTile(out) || a != b)
    return op->emitOpError(
        "requires matching one-dimensional scalable integer input tiles");
  StringRef name = op->getName().getStringRef();
  unsigned width = a.getElementTypeBitWidth();
  unsigned destWidth = out.getElementTypeBitWidth();
  bool wide = name == "ime.intr.vpack" || name == "ime.intr.vupack";
  bool nibble = name.ends_with("4");
  if (wide) {
    if ((width != 8 && width != 16 && width != 32 && width != 64) ||
        a.getNumElements() * width != 64 || destWidth != width ||
        out.getNumElements() != 2 * a.getNumElements())
      return op->emitOpError("pack/upack requires one-register inputs and a "
                             "same-width register-pair result");
  } else if (nibble) {
    if (width != 8 || a.getNumElements() != 8 || out != a)
      return op->emitOpError(
          "4-bit pack requires vector<[8]xi8> inputs and result");
  } else if ((width != 16 && width != 32 && width != 64) ||
             a.getNumElements() * width != 64 || destWidth * 2 != width ||
             out.getNumElements() != 2 * a.getNumElements()) {
    return op->emitOpError("narrow pack requires one-register inputs and a "
                           "result with twice the lanes and half the width");
  }
  return success();
}
} // namespace

LogicalResult Vmadot_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadotu_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadotsu_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadotus_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vfmadot_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot1_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot1u_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot1su_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot1us_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot2_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot2u_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot2su_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot2us_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot3_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot3u_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot3su_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot3us_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadotn_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadotnu_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadotnsu_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadotnus_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vfmadot1_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vfmadot2_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vfmadot3_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vfmadotn_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vpack_IntrOp::verify() { return verifyA100(getOperation()); }

LogicalResult Vupack_IntrOp::verify() { return verifyA100(getOperation()); }

LogicalResult Vnpack_IntrOp::verify() { return verifyA100(getOperation()); }

LogicalResult Vnspack_IntrOp::verify() { return verifyA100(getOperation()); }

LogicalResult Vnpack4_IntrOp::verify() { return verifyA100(getOperation()); }

LogicalResult Vnspack4_IntrOp::verify() { return verifyA100(getOperation()); }
