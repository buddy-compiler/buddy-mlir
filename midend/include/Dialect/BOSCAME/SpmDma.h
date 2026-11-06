//===- SpmDma.h - SPM / DMA platform contract and helpers ------===//
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
// The patch7 / nr-fpga SPM and DMA hardware contract, shared by the BOSCAME
// dialect verifiers and by the SPM/DMA passes.
//
// Everything in this file is a *hardware* constant: bank sizes and bases, DMA
// register-block bases and offsets, alignment granularity.  Keeping them in one
// place is what lets the verifiers, the check pass and the lowering agree on a
// single contract.  These are not tuning parameters of a particular model.
//
//===----------------------------------------------------------------------===//

#ifndef BOSCAME_SPMDMA_H
#define BOSCAME_SPMDMA_H

#include "BOSCAME/BOSCAMEAttrs.h"
#include "BOSCAME/BOSCAMETypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

#include <optional>

namespace buddy {
namespace boscame {
namespace spmdma {

//===----------------------------------------------------------------------===//
// SPM
//===----------------------------------------------------------------------===//

// Number of SPM banks.  Banks are storage, not DMA engines: bank 0 lives in
// [0x20000000, 0x20020000) and bank 1 in [0x20400000, 0x20420000).
constexpr int64_t kSpmBankCount = 2;
constexpr int64_t kSpmBankCapacityBytes = 128 * 1024;
constexpr uint64_t kSpmBankBaseAddress[kSpmBankCount] = {0x20000000ULL,
                                                         0x20400000ULL};

//===----------------------------------------------------------------------===//
// DMA
//===----------------------------------------------------------------------===//

// Number of DMA engines.  DMA channels are transfer engines, not storage:
// channel 0 is controlled through 0x60000000 and channel 1 through
// 0x60010000.  A bank and a channel are orthogonal, so DMA0 may move data
// into or out of SPM1.
constexpr int64_t kDmaChannelCount = 2;
constexpr uint64_t kDmaChannelBaseAddress[kDmaChannelCount] = {0x60000000ULL,
                                                               0x60010000ULL};

// Every control/status register is a natural-aligned single-shot 64-bit access
// and every transfer offset and length is a multiple of 64 bytes.
constexpr int64_t kDmaAlignmentBytes = 64;

// Register offsets inside a DMA channel block.
constexpr uint64_t kDmaSrcRegisterOffset = 0x00;
constexpr uint64_t kDmaDstRegisterOffset = 0x08;
constexpr uint64_t kDmaLenRegisterOffset = 0x10;
constexpr uint64_t kDmaStartRegisterOffset = 0x18;
constexpr uint64_t kDmaStatusRegisterOffset = 0x20;

// `STATUS` bits.  The value is live, not sticky: `busy` is set while the
// channel has an outstanding request.
constexpr uint64_t kDmaStatusBusyBit = 1 << 0;
constexpr uint64_t kDmaStatusBusErrorBit = 1 << 1;

// Names of the two runtime helpers that `bosc_ame.dma.start` /
// `bosc_ame.dma.wait` are lowered to.  Both take an `i32` channel so that the
// runtime can select the register block; `bosc_ame_dma_wait` additionally
// receives the original effective SRC address (the token) so it can re-issue
// the config write that acts as the completion barrier, then checks `STATUS`
// and takes the runtime error path on failure.  It returns `void`: a normal
// return already means "completed and succeeded".
inline constexpr llvm::StringLiteral kDmaStartHelperName = "bosc_ame_dma_start";
inline constexpr llvm::StringLiteral kDmaWaitHelperName = "bosc_ame_dma_wait";

// The kernel ABI assertion a module must carry when a DMA endpoint is not SPM
// storage.  The compiler cannot see the callers of an entry kernel, so the
// alignment of an ordinary pointer argument and the disjointness of two
// pointer arguments are caller-owned facts.  `args_aligned64_nonalias` is the
// explicit statement that every pointer argument is 64-byte aligned and that
// distinct pointer arguments do not overlap.
//
// `args_aligned64` asserts only the alignment. Potentially overlapping DDR
// ranges still require ordering or a separate disjointness proof. Without
// either assertion, an unproven DDR endpoint alignment is rejected.
inline constexpr llvm::StringLiteral kDmaAbiAttrName = "bosc_ame.dma.abi";
inline constexpr llvm::StringLiteral kDmaAbiAlignedNonAliasing =
    "args_aligned64_nonalias";
// Alignment alone suffices for a synchronous DDR/SPM boundary. This does
// not establish disjointness between DDR arguments; the lifetime checker
// continues treating those roots as potentially aliasing.
inline constexpr llvm::StringLiteral kDmaAbiAligned = "args_aligned64";

//===----------------------------------------------------------------------===//
// Type queries
//===----------------------------------------------------------------------===//

// Byte width of the element types an SPM `memref` is allowed to use.  Returns
// 0 for everything else (i1, i4, f8, i128, nested memrefs, ...) so that
// `product(shape) * width` is never ambiguous.
inline unsigned getSpmElementByteWidth(mlir::Type elementType) {
  if (auto intType = llvm::dyn_cast<mlir::IntegerType>(elementType)) {
    switch (intType.getWidth()) {
    case 8:
    case 16:
    case 32:
    case 64:
      return intType.getWidth() / 8;
    default:
      return 0;
    }
  }
  if (auto floatType = llvm::dyn_cast<mlir::FloatType>(elementType)) {
    switch (floatType.getWidth()) {
    case 16: // f16 and bf16
      return 2;
    case 32:
      return 4;
    case 64:
      return 8;
    default:
      return 0;
    }
  }
  return 0;
}

// Bank of a memref type whose memory space is `#bosc_ame.spm<bank>`.
// Returns std::nullopt for anything else, including DDR memrefs.
inline std::optional<int64_t> getSpmBank(mlir::Type type) {
  auto memref = llvm::dyn_cast<mlir::BaseMemRefType>(type);
  if (!memref)
    return std::nullopt;
  auto space = llvm::dyn_cast_or_null<::buddy::boscame::SPMAttr>(
      memref.getMemorySpace());
  if (!space)
    return std::nullopt;
  return space.getBank();
}

// True when `shape` is addressed by `strides` with element stride equal to the
// element width, i.e. the buffer is a single contiguous run of bytes exactly
// `getStaticByteSize(...)` long.
inline bool hasContiguousStrides(llvm::ArrayRef<int64_t> shape,
                                 llvm::ArrayRef<int64_t> strides) {
  if (shape.size() != strides.size())
    return false;
  int64_t expected = 1;
  for (size_t index = shape.size(); index-- > 0;) {
    if (strides[index] != expected)
      return false;
    if (llvm::MulOverflow(expected, shape[index], expected))
      return false;
  }
  return true;
}

// Byte size of a statically shaped, contiguously laid out buffer, or
// std::nullopt when the size cannot be computed (dynamic shape, unsupported
// element type, non-contiguous layout, overflow).
inline std::optional<int64_t> getStaticByteSize(mlir::ShapedType shaped) {
  if (!shaped.hasStaticShape())
    return std::nullopt;
  unsigned elementBytes = getSpmElementByteWidth(shaped.getElementType());
  if (elementBytes == 0)
    return std::nullopt;
  int64_t elementCount = 1;
  for (int64_t dim : shaped.getShape()) {
    if (dim < 0 || llvm::MulOverflow(elementCount, dim, elementCount))
      return std::nullopt;
  }
  int64_t totalBytes = 0;
  if (llvm::MulOverflow(elementCount, static_cast<int64_t>(elementBytes),
                        totalBytes))
    return std::nullopt;
  return totalBytes;
}

// Byte size of a ranked memref when it is contiguous and has a zero offset.
// Unlike `getStaticByteSize` this rejects subviews with a nonzero descriptor
// offset, which is what a `bosc_ame.spm.view` result guarantees and what lets
// the view be lowered to a single fixed base address.
inline std::optional<int64_t>
getContiguousZeroOffsetByteSize(mlir::MemRefType memref) {
  llvm::SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (mlir::failed(memref.getStridesAndOffset(strides, offset)))
    return std::nullopt;
  if (offset != 0)
    return std::nullopt;
  if (!hasContiguousStrides(memref.getShape(), strides))
    return std::nullopt;
  return getStaticByteSize(memref);
}

//===----------------------------------------------------------------------===//
// DMA endpoints
//===----------------------------------------------------------------------===//

// A DMA endpoint is a typed contiguous view of a buffer. The hardware moves
// `bytes` bytes from the effective SRC address to the effective DST address,
// so an endpoint must be a 1-D ranked memref with a supported byte-addressable
// element type, unit element stride and static shape. A dynamic descriptor
// offset needs an SSA alignment proof. The descriptor offset stays part of the
// effective address; it is never silently dropped.
struct DmaEndpoint {
  mlir::MemRefType type;
  // SPM bank when the memory space is `#bosc_ame.spm<bank>`, nullopt for DDR.
  std::optional<int64_t> bank;
  // Static descriptor offset in bytes inside the addressed storage.
  int64_t byteOffset = 0;
  bool hasStaticOffset = true;
  // Static length of the endpoint buffer in bytes.
  int64_t byteLength = 0;

  bool isSpm() const { return bank.has_value(); }
};

// Structural endpoint check shared by the op verifier and the check pass.
inline std::optional<DmaEndpoint> getDmaEndpoint(mlir::MemRefType type) {
  if (!type || type.getRank() != 1)
    return std::nullopt;
  unsigned elementBytes = getSpmElementByteWidth(type.getElementType());
  if (!elementBytes)
    return std::nullopt;
  if (!type.hasStaticShape())
    return std::nullopt;
  llvm::SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (mlir::failed(type.getStridesAndOffset(strides, offset)))
    return std::nullopt;
  if (!hasContiguousStrides(type.getShape(), strides))
    return std::nullopt;
  DmaEndpoint endpoint;
  endpoint.type = type;
  endpoint.bank = getSpmBank(type);
  endpoint.hasStaticOffset = offset != mlir::ShapedType::kDynamic;
  if (endpoint.hasStaticOffset &&
      llvm::MulOverflow(offset, static_cast<int64_t>(elementBytes),
                        endpoint.byteOffset))
    return std::nullopt;
  if (llvm::MulOverflow(type.getDimSize(0), static_cast<int64_t>(elementBytes),
                        endpoint.byteLength))
    return std::nullopt;
  return endpoint;
}

// Effective address alignment of an endpoint, as far as it is statically
// known.  SPM storage is anchored at a 64-byte aligned bank base, so the
// descriptor offset alone decides.  A DDR base address is not visible to the
// compiler and is covered by `bosc_ame.dma.abi` instead.
inline bool isStaticallyAlignedEndpoint(const DmaEndpoint &endpoint) {
  return endpoint.hasStaticOffset &&
         endpoint.byteOffset % kDmaAlignmentBytes == 0;
}

} // namespace spmdma
} // namespace boscame
} // namespace buddy

#endif // BOSCAME_SPMDMA_H
