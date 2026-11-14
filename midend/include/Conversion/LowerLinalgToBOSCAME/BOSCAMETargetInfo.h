//===- BOSCAMETargetInfo.h - BOSCAME target capability profile --------===//
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
// Static capability profile consulted by BOSCAME planning and legality checks.
// Values come from dialect constants and the named target table; unmeasured
// contracts (for example RVV width) stay optional instead of being guessed.
//
// This header describes storage endpoints, DMA transfer limits, AME tile
// layouts/roles, and allocator-facing register budgets. It does not encode
// schedules, cost models, or placement decisions.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMETARGETINFO_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMETARGETINFO_H

#include "Dialect/BOSCAME/SpmDma.h"
#include "Dialect/BOSCAME/TileOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <optional>

namespace buddy::boscame {

// Logical memory endpoint used when stating transfer direction support.
enum class StorageKind { DDR, SPM };

// DMA / transfer limits between storage endpoints for the selected target.
struct TransferCapability {
  int64_t endpointAlignmentBytes = 0;
  int64_t packetGranularityBytes = 0;
  int64_t maxStaticRank = 0;
  bool supportsDdrToSpm = false;
  bool supportsSpmToDdr = false;
  bool supportsSpmToSpm = false;
  bool packetizesPitchedRows = false;

  // True when the profile permits a transfer from `source` to `destination`.
  constexpr bool supports(StorageKind source, StorageKind destination) const {
    if (source == StorageKind::DDR && destination == StorageKind::SPM)
      return supportsDdrToSpm;
    if (source == StorageKind::SPM && destination == StorageKind::DDR)
      return supportsSpmToDdr;
    if (source == StorageKind::SPM && destination == StorageKind::SPM)
      return supportsSpmToSpm;
    return false;
  }
};

// Contiguous stride conventions accepted by AME tile operands.
enum class AmeTileLayout {
  // Logical A[M, K] with K contiguous.
  MxKWithKContiguous,
  // Logical B[K, N] backed by physical [N, K] storage with K contiguous.
  KxNWithPhysicalNxK,
  // Accumulator[M, N] with N contiguous.
  MxNWithNContiguous,
};

// Stable string token for diagnostics and plan reports.
inline llvm::StringRef stringifyAmeTileLayout(AmeTileLayout layout) {
  switch (layout) {
  case AmeTileLayout::MxKWithKContiguous:
    return "mk-k-contiguous";
  case AmeTileLayout::KxNWithPhysicalNxK:
    return "kn-physical-nk-k-contiguous";
  case AmeTileLayout::MxNWithNContiguous:
    return "mn-n-contiguous";
  }
  llvm_unreachable("unknown AME tile layout");
}

// True when a static-shape rank-2 memref matches the layout's stride rules.
inline bool hasAmeTileLayout(mlir::MemRefType type, AmeTileLayout layout) {
  if (!type || type.getRank() != 2 || !type.hasStaticShape())
    return false;
  llvm::SmallVector<int64_t, 2> strides;
  int64_t offset = 0;
  if (mlir::failed(type.getStridesAndOffset(strides, offset)) ||
      strides.size() != 2 || strides[0] <= 0 || strides[1] <= 0)
    return false;
  switch (layout) {
  case AmeTileLayout::MxKWithKContiguous:
  case AmeTileLayout::MxNWithNContiguous:
    return strides[1] == 1 && strides[0] >= type.getDimSize(1);
  case AmeTileLayout::KxNWithPhysicalNxK:
    return strides[0] == 1 && strides[1] >= type.getDimSize(0);
  }
  llvm_unreachable("unknown AME tile layout");
}

// Per-role AME operand constraints (A, B, or accumulator).
struct AmeRoleCapability {
  unsigned elementBits = 0;
  AmeTileLayout layout = AmeTileLayout::MxKWithKContiguous;
  bool supportsSpm = false;
  bool supportsDirectDdr = false;
};

// The marked-matmul contract implemented by the AME dialect and lowering.
// Integer signedness is selected by the matmul's signed cast; memref i8 types
// themselves are signless in MLIR.
struct AmeCapability {
  int64_t maxTileM = 0;
  int64_t maxTileN = 0;
  int64_t maxTileK = 0;
  bool signedIntegerMultiply = false;
  AmeRoleCapability a;
  AmeRoleCapability b;
  AmeRoleCapability accumulator;
};

// Hardware facts known to the BOSCAME dialect boundary. Unmeasured contracts
// such as RVV width remain unavailable instead of being inferred.
struct TargetInfo {
  llvm::StringRef name;
  int64_t spmBankCount = 0;
  int64_t spmBankCapacityBytes = 0;
  int64_t dmaChannelCount = 0;
  AmeCapability ame;
  std::optional<int64_t> rvvVectorBytes;
  TransferCapability dma;
  // Capacity supported by the current allocator/encoder, not a speculative
  // hardware register count. Unknown RVV width remains independent.
  int64_t allocatableTileRegisters = 8;
  int64_t allocatableAccumulatorRegisters = 8;
  int64_t scalarFloatRegisters = 32;
};

// Built-in `nr-fpga` profile assembled from dialect SPM/DMA/AME constants.
// RVV width remains unset (`nullopt`) because it is unmeasured here.
inline constexpr TargetInfo getNrFpgaTargetInfo() {
  return {"nr-fpga",
          spmdma::kSpmBankCount,
          spmdma::kSpmBankCapacityBytes,
          spmdma::kDmaChannelCount,
          {kAmeMaxTileDim,
           kAmeMaxTileDim,
           kAmeMaxTileDim,
           true,
           {8, AmeTileLayout::MxKWithKContiguous, true, true},
           {8, AmeTileLayout::KxNWithPhysicalNxK, true, true},
           {32, AmeTileLayout::MxNWithNContiguous, true, true}},
          std::nullopt,
          {spmdma::kDmaAlignmentBytes, spmdma::kDmaAlignmentBytes, 2, true,
           true, true, true}};
}

// Resolve a named profile. Empty names default to `nr-fpga`; unknown names
// return nullopt so callers can reject unsupported targets explicitly.
inline std::optional<TargetInfo> resolveTargetInfo(llvm::StringRef name) {
  if (name.empty() || name == "nr-fpga")
    return getNrFpgaTargetInfo();
  return std::nullopt;
}

} // namespace buddy::boscame

#endif // BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMETARGETINFO_H
