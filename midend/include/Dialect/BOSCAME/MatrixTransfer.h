//===- MatrixTransfer.h - Exact NR matrix-register copies ------===//
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
// Exact ACC32 matrix-register load/store contracts for the NR FPGA path.
// Encodes the closed subset already exercised by byte-preservation tests:
// instruction width, register class, supported element types, and layout
// helpers shared by tile op verification and LLVM intrinsic lowering.
//
// This is not a claim about implementation-independent MLEN/RLEN/AMUL
// capacities or other memory modes. Formats outside the closed set (including
// f16/bf16) need explicit validation before being enabled.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_DIALECT_BOSCAME_MATRIXTRANSFER_H
#define BUDDY_DIALECT_BOSCAME_MATRIXTRANSFER_H

#include "Dialect/BOSCAME/SpmDma.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

#include <optional>
#include <utility>

namespace buddy::boscame::matrixtransfer {

// Bounded NR ACC32 subset constants shared by verifiers and lowering.
inline constexpr int64_t kInstructionBits = 32;
inline constexpr int64_t kMaxRows = 64;
inline constexpr int64_t kMaxRowBytes = 256;
inline constexpr int64_t kRegisterBytes = 16384;
inline constexpr int64_t kRegisterCount = 1;
inline constexpr int64_t kMtype = 65602;
inline constexpr llvm::StringLiteral kRegisterClass = "acc";

// Closed logical storage types covered by the raw-word copy contract and
// byte-preservation tests. Logical width does not select the instruction
// width: every accepted type uses the same ACC32 load/store bundle. Other
// formats, including f16/bf16, need explicit validation before being enabled.
inline bool isSupportedElementType(mlir::Type type) {
  if (!type)
    return false;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return integer.isSignless() &&
           (integer.getWidth() == 8 || integer.getWidth() == 16 ||
            integer.getWidth() == 32 || integer.getWidth() == 64);
  return type.isF32() || type.isF64();
}

// A nonoverlapping rectangular view in physical row order. All sizes are
// bytes except rows and unitAxis. The descriptor offset is deliberately not
// included: copy offsets are relative to the endpoint's effective pointer.
struct Layout {
  unsigned unitAxis;
  int64_t rows;
  int64_t rowBytes;
  int64_t pitch;
  int64_t elementBytes;
};

inline std::optional<Layout> getLayout(mlir::MemRefType type,
                                       unsigned unitAxis) {
  if (!type || type.getRank() != 2 || !type.hasStaticShape() || unitAxis > 1 ||
      !isSupportedElementType(type.getElementType()))
    return std::nullopt;
  int64_t width = spmdma::getSpmElementByteWidth(type.getElementType());
  llvm::SmallVector<int64_t, 2> strides;
  int64_t offset;
  if (!width || mlir::failed(type.getStridesAndOffset(strides, offset)) ||
      strides[unitAxis] != 1 || strides[1 - unitAxis] <= 0 ||
      type.getDimSize(0) <= 0 || type.getDimSize(1) <= 0 ||
      (offset != mlir::ShapedType::kDynamic && offset < 0))
    return std::nullopt;
  Layout layout{unitAxis, type.getDimSize(1 - unitAxis), 0, 0, width};
  if (llvm::MulOverflow(type.getDimSize(unitAxis), width, layout.rowBytes) ||
      llvm::MulOverflow(strides[1 - unitAxis], width, layout.pitch) ||
      (layout.rows > 1 && layout.pitch < layout.rowBytes))
    return std::nullopt;
  return layout;
}

inline std::optional<unsigned> getCommonUnitAxis(mlir::MemRefType source,
                                                 mlir::MemRefType target) {
  if (!source || !target || source.getElementType() != target.getElementType())
    return std::nullopt;
  // Prefer the ordinary trailing contiguous axis for degenerate layouts.
  for (unsigned axis : {1u, 0u})
    if (getLayout(source, axis) && getLayout(target, axis))
      return axis;
  return std::nullopt;
}

// Validate one exact subrectangle and recover its physical row/byte-column
// origin. No segment may start in or run through a pitched row's padding.
inline std::optional<std::pair<int64_t, int64_t>>
getBlockOrigin(const Layout &layout, int64_t offset, int64_t rows,
               int64_t rowBytes, int64_t pitch) {
  if (offset < 0 || rows <= 0 || rows > kMaxRows || rowBytes <= 0 ||
      rowBytes > kMaxRowBytes || rowBytes % 4 || offset % 4 || pitch <= 0 ||
      pitch % 4 || pitch != layout.pitch || offset % layout.elementBytes ||
      rowBytes % layout.elementBytes)
    return std::nullopt;
  int64_t row = layout.rows == 1 ? 0 : offset / pitch;
  int64_t column = layout.rows == 1 ? offset : offset % pitch;
  int64_t lastRow, lastColumn, bytes, lastByte;
  if (llvm::AddOverflow(row, rows, lastRow) || lastRow > layout.rows ||
      llvm::AddOverflow(column, rowBytes, lastColumn) ||
      lastColumn > layout.rowBytes ||
      llvm::MulOverflow(rows, rowBytes, bytes) || bytes > kRegisterBytes ||
      llvm::MulOverflow(rows - 1, pitch, lastByte) ||
      llvm::AddOverflow(lastByte, offset, lastByte) ||
      llvm::AddOverflow(lastByte, rowBytes, lastByte))
    return std::nullopt;
  return std::pair<int64_t, int64_t>(row, column);
}

} // namespace buddy::boscame::matrixtransfer

#endif // BUDDY_DIALECT_BOSCAME_MATRIXTRANSFER_H
