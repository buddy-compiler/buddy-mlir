//===- BOSCAMEAccumulatorPlacement.h - Measured hot-P policy --------------===//
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
// Measured hot-accumulator placement policy: compare DDR vs SPM total cycle
// budgets (half-cycle integer units with evidence hashes) and emit placement
// when the closed HotAccumulator proof plus MeasuredAccumulator profile match.
// Timings are complete-adapter totals, not per-byte coefficients.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEACCUMULATORPLACEMENT_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEACCUMULATORPLACEMENT_H

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEMeasuredAccumulator.h"
#include "mlir/IR/Builders.h"

namespace buddy::boscame {

// Complete typed-adapter timings, in half-cycle integer units. The lower
// paired DDR median and higher SPM median across both input stresses bound
// the observed comparison conservatively. The margin is the largest observed
// DDR/SPM range, including all DDR control calls. These are measured totals,
// not per-byte coefficients: frame, fill, copies, epilogue and fences are
// already included and must not be charged again.
struct AccumulatorTiming {
  int64_t ddrCyclesX2, spmCyclesX2, marginCyclesX2;
  llvm::StringRef evidence;
};

// Look up the published timing envelope for a closed measured adapter shape.
inline AccumulatorTiming accumulatorTiming(MeasuredAccumulatorMode mode,
                                           int64_t trips) {
  if (mode == MeasuredAccumulatorMode::Overwrite) {
    if (trips == 6)
      return {1275161, 177632, 980, "573a4d65db6847b690ee5c29f37b4f0b"};
    return {1628473, 556539, 1712,
            "3217f5dc739740a58ef3e3a13ef40240+"
            "a82c64e45ec0473c8368bd10e62b2674"};
  }
  if (trips == 6)
    return {2453211, 677892, 814, "1cf30d811eaa4bcb9f947041d3d18f0b"};
  return {2807472, 1047294, 1314,
          "6a7ff0d0ec1d40cd9d42c3e6c2a43df4+"
          "6c0b5c40159246488f0afd8c50bca1b2"};
}

// Canonical optional placement, independently reconstructed by verifier.
// No serialized plan field is consulted. ACC/stream/float ownership and
// explicit target windows are resolved before calling this function. The
// first measured envelope is one closed adapter at bank0/offset0 without
// any other fixed root; reserve every target constraint first, including
// windows later in source order or in bank1 (unmeasured epilogue context).
inline mlir::DictionaryAttr getHotAccumulatorPlacement(mlir::Value storage,
                                                       bool automatic,
                                                       bool supportedWindow) {
  auto hot = analyzeHotAccumulator(storage);
  if (!hot)
    return {};
  mlir::Builder builder(storage.getContext());
  mlir::NamedAttrList fields;
  auto integer = [&](llvm::StringRef key, int64_t value) {
    fields.set(key, builder.getI64IntegerAttr(value));
  };
  auto string = [&](llvm::StringRef key, llvm::StringRef value) {
    fields.set(key, builder.getStringAttr(value));
  };
  string("level", "ddr");
  string("role", "hot-accumulator");
  integer("alignment", 64);
  integer("capacity", hot->bytes);
  integer("trips", hot->trips);
  integer("loop_bytes", hot->loopBytes);
  integer("boundary_bytes", hot->boundaryBytes);
  // Declining an unavailable window needs no extrapolation from the timing
  // table. Extra fixed-buffer computation is otherwise outside the measured
  // whole-function source closure and cannot qualify a positive selection.
  if (!automatic) {
    string("reason", "hot-accumulator-debug-disabled");
  } else if (!supportedWindow) {
    string("reason", "hot-accumulator-no-supported-window");
  } else if (auto profile = analyzeMeasuredAccumulator(storage)) {
    auto timing = accumulatorTiming(profile->mode, profile->trips);
    int64_t saving = 0;
    bool profitable =
        !llvm::SubOverflow(timing.ddrCyclesX2, timing.spmCyclesX2, saving) &&
        saving > timing.marginCyclesX2;
    string("profile", "nr-e6-hot-p-v1");
    string("evidence", timing.evidence);
    string("mode", profile->mode == MeasuredAccumulatorMode::Overwrite
                       ? "overwrite"
                       : "normal-post-add");
    integer("pitch", profile->pitch);
    integer("ddr_cycles_x2", timing.ddrCyclesX2);
    integer("spm_cycles_x2", timing.spmCyclesX2);
    integer("margin_cycles_x2", timing.marginCyclesX2);
    integer("saving_cycles_x2", saving);
    string("reason", profitable ? "hot-accumulator-measured-profitable"
                                : "hot-accumulator-benefit-within-margin");
    if (profitable) {
      string("level", "spm");
      integer("bank", 0);
      integer("offset", 0);
    }
  } else {
    string("reason", "hot-accumulator-no-measured-profile");
  }
  return fields.getDictionary(storage.getContext());
}

} // namespace buddy::boscame
#endif
