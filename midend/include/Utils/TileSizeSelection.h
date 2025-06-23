//====- TileSizeSelection.h
//-----------------------------------------------------------===//
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
//===------------------------------------------------------------------------===//
//
// This file defines tiling utils functions for the buddy compiler
// ecosystem.
//
//===-----------------------------------------------------------------------===//
#ifndef INCLUDE_UTILS_TILESIZESELECTION_H_
#define INCLUDE_UTILS_TILESIZESELECTION_H_

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

using namespace mlir;

namespace buddy {
// Map a range to a SmallVector with element types deduced from the mapping.
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