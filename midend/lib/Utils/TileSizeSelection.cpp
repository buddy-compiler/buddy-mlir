//====- TileSizeSelection.cpp
//---------------------------------------------------------===//
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
// This file implements tiling utils functions for the buddy compiler
// ecosystem.
//
//===----------------------------------------------------------------------===//
#include "Utils/TileSizeSelection.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace buddy {
void setSCFTileSizes(scf::SCFTilingOptions &options, TilingInterface consumerOp,
                     SmallVector<int64_t> tileSizes,
                     SmallVector<bool> tileScalableFlags) {
  int numLoops = consumerOp.getLoopIteratorTypes().size();
  tileSizes.resize(numLoops, 0);
  tileScalableFlags.resize(numLoops, false);
  if (!llvm::is_contained(tileSizes, 1)) {
    // Non-scalabe case: All constant tile sizes.
    options.setTileSizes(tileSizes);
    // getAsIndexOpFoldResult(consumerOp.getContext(), tileSizes));
  } else {
    // Scalable case: Multiply scalable tile sizes by a vector.vscale op.
    options.setTileSizeComputationFunction(
        [=](OpBuilder &builder, Operation *op) -> SmallVector<Value> {
          auto loc = op->getLoc();
          return map_to_vec(
              llvm::zip(tileSizes, tileScalableFlags), [&](auto pair) -> Value {
                auto [t, isScalable] = pair;
                Value size = builder.create<arith::ConstantIndexOp>(loc, t);
                if (isScalable) {
                  Value vscale = builder.create<vector::VectorScaleOp>(loc);
                  size = builder.create<arith::MulIOp>(loc, size, vscale);
                }
                return size;
              });
        });
  }
}
} // namespace buddy