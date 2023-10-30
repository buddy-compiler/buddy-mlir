#ifndef INCLUDE_UTILS_TILESIZESELECTION_H_
#define INCLUDE_UTILS_TILESIZESELECTION_H_

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

using namespace mlir;

namespace buddy {
// Map a range to a SmallVectot with element types deduced from the mapping.
template <unsigned Size, class ContainerTy, class FuncTy>
auto map_to_vec(ContainerTy &&C, FuncTy &&F) {
  return llvm::to_vector<Size>(
      llvm::map_range(std::forward<ContainerTy>(C), std::forward<FuncTy>(F)));
}

template <class ContainerTy, class FuncTy>
auto map_to_vec(ContainerTy &&C, FuncTy &&F) {
  return to_vector(
      map_range(std::forward<ContainerTy>(C), std::forward<FuncTy>(F)));
}

void setSCFTileSizes(scf::SCFTilingOptions &options, TilingInterface consumerOp,
                     SmallVector<int64_t> tileSizes,
                     SmallVector<bool> tileScalableFlags);
} // namespace buddy

#endif // INCLUDE_UTILS_TILESIZESELECTION_H_