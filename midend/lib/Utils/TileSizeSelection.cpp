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
    options.setTileSizes(
        tileSizes);
    // getAsIndexOpFoldResult(consumerOp.getContext(), tileSizes));
  } else {
    // Scalable case: Multiply scalable tile sizes by a vector.vscale op.
    options.setTileSizeComputationFunction(
        [=](OpBuilder &builder, Operation *op) -> SmallVector<Value> {
          auto loc = op->getLoc();
          return map_to_vec(
              llvm::zip(tileSizes, tileScalableFlags),
              [&](auto pair) -> Value {
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