//===-- Tile.td - Tile dialect operation definitions --- tablegen --===//
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
// This file defines the basic operations for the Tile dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TILE_OPS
#define TILE_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Tile dialect definition
//===----------------------------------------------------------------------===//

def Tile_Dialect : Dialect {
  let name = "tile";
  let cppNamespace = "::buddy::tile";
  let summary = "A dialect for multi-level tiling operations";
  let description = [{
    The Tile dialect provides operations for hierarchical matrix tiling and 
    computation. It serves as an intermediate abstraction between high-level 
    linear algebra operations (Linalg) and low-level hardware-specific 
    operations (Buckyball).
    
    The dialect supports a multi-level tiling hierarchy:
    - VecTile: Top-level tile operations
    - MergeTile: Mid-level tile operations  
    - MetaTile: Fine-grained tile operations
  }];
}

//===----------------------------------------------------------------------===//
// Tile operation definitions
//===----------------------------------------------------------------------===//

class Tile_Op<string mnemonic, list<Trait> traits = []> :
  Op<Tile_Dialect, mnemonic, traits> {}

/* Tiling Design:
 *  The Tile dialect is responsible for partitioning arbitrary-sized matrices
 *  into fixed-size computation tiles that match the hardware constraints.
 *  
 *  TileMatMulOp:
 *  - Input: Arbitrary-sized matrices A[M][K], B[K][N], C[M][N]
 *  - Output: Tiled matrix multiplication that fits hardware SPAD/ACC constraints
 *  - This operation will be lowered to Buckyball MatMulOp operations
 */

def TileMatMulOp : Tile_Op<"tile_matmul"> {
  let summary = "Tiled matrix multiplication";
  let description = [{
    Perform matrix multiplication with automatic tiling to fit hardware constraints.
    This operation takes arbitrary-sized input matrices and tiles them into 
    fixed-size blocks that can be efficiently executed on the accelerator.
    
    The operation handles:
    - Computation of optimal tile sizes based on SPAD and accumulator constraints
    - Padding for cases where matrix dimensions are not perfectly divisible
    - Generation of tiled computation loops
    
    This will be lowered to Buckyball MatMulOp operations with appropriate
    data movement (mvin/mvout) operations.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$aMemArray,
                       MemRefRankOf<[AnyType], [2]>:$bMemArray,
                       MemRefRankOf<[AnyType], [2]>:$cMemArray);
  let assemblyFormat = [{$aMemArray $bMemArray $cMemArray attr-dict `:` 
    type($aMemArray) type($bMemArray) type($cMemArray)}];
}

def TileTransposeOp : Tile_Op<"tile_transpose"> {
  let summary = "Tiled matrix transpose";
  let description = [{
    Transpose a matrix with automatic tiling to fit hardware constraints.
    This operation takes an arbitrary-sized input matrix and tiles it into 
    fixed-size blocks that can be efficiently executed on the accelerator.
    
    The operation handles:
    - Computation of optimal tile sizes based on SPAD and accumulator constraints
    - Padding for cases where matrix dimensions are not perfectly divisible
    - Generation of tiled computation loops
    
    This will be lowered to Buckyball TransposeOp operations with appropriate
    data movement (mvin/mvout) operations.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$aMemArray,
                       MemRefRankOf<[AnyType], [2]>:$bMemArray);
  let assemblyFormat = [{$aMemArray $bMemArray attr-dict `:` 
    type($aMemArray) type($bMemArray)}];
}

#endif // TILE_OPS

