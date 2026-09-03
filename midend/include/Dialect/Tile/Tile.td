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
  let summary = "A dialect for tiled ops that lower to Buckyball";
  let description = [{
    Tile ops (`tile_matmul`, `tile_transpose`, `tile_conv2d`) take whole
    memrefs and are lowered by `convert-tile-to-buckyball` into `scf` loops
    plus `memref.subview` tiles and per-tile Buckyball ops (`bb_matmul`, etc.).
    There is no separate MergeTile/MetaTile dialect layer; tiling depth is
    decided inside that pass from the 16-byte bank row, 16xK matrix ISA tile,
    and bank depth limits, not as distinct ops here.
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

def TileMegaMatmulOp : Tile_Op<"mega_matmul"> {
  let summary = "INT8 MatMul stage inside a Tile MegaKernel";
  let description = [{
    INT8 activation and weight, offline INT32 bias, activation LUT, and
    per-output-channel scaling form one stage of a resident kernel.
  }];
  let arguments = (ins MemRefRankOf<[I8], [2]>:$input,
                       MemRefRankOf<[I8], [2]>:$weight,
                       MemRefRankOf<[I32], [1]>:$bias,
                       MemRefRankOf<[F32], [1]>:$scale,
                       MemRefRankOf<[I8], [1]>:$lut,
                       MemRefRankOf<[I8, F32], [2]>:$output,
                       I64Attr:$activation,
                       F32Attr:$outputScale);
  let assemblyFormat = [{$input $weight $bias $scale $lut $output attr-dict `:`
    type($input) type($weight) type($bias) type($scale) type($lut) type($output)}];
}

def TileQuantF32ToI8Op : Tile_Op<"quant_f32_to_i8"> {
  let summary = "Per-tensor model input quantization";
  let arguments = (ins MemRefRankOf<[F32], [2, 4]>:$input,
                       MemRefRankOf<[I8], [2, 4]>:$output,
                       F32Attr:$scale,
                       BoolAttr:$nchwToNhwc);
  let assemblyFormat = [{$input $output attr-dict `:` type($input) type($output)}];
}

def TileMegaYieldOp : Tile_Op<"mega_yield", [Terminator]> {
  let summary = "Terminate a Tile MegaKernel region";
  let assemblyFormat = "attr-dict";
}

def TileMegaKernelOp : Tile_Op<"mega_kernel", [
    SingleBlockImplicitTerminator<"TileMegaYieldOp">]> {
  let summary = "One cross-layer resident quantized kernel";
  let arguments = (ins AnyMemRef:$input,
                       AnyMemRef:$output);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{$input $output attr-dict `:`
    type($input) type($output) $body}];
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

def TileReluOp : Tile_Op<"tile_relu"> {
  let summary = "Tiled signed INT32 ReLU";
  let description = [{
    Perform D = max(X, 0) on an arbitrary-sized INT32 matrix. Lowering pads
    both dimensions to 16, then runs 512x16 bank tiles.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input,
                       MemRefRankOf<[AnyType], [2]>:$output);
  let assemblyFormat = [{$input $output attr-dict `:`
    type($input) type($output)}];
}

def TileConv2dOp : Tile_Op<"tile_conv2d"> {
  let summary = "Tiled 2D convolution (NHWC layout)";
  let description = [{
    Perform 2D convolution with automatic tiling.
    Input: [N, H, W, C], Filter: [KH, KW, C, OC], Output: [N, OH, OW, OC].
    Optional bb_pad_* attrs encode fused tensor.pad (zero).
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [4]>:$input,
                       MemRefRankOf<[AnyType], [4]>:$filter,
                       MemRefRankOf<[AnyType], [4]>:$output,
                       DefaultValuedAttr<I64Attr, "0">:$padLow,
                       DefaultValuedAttr<I64Attr, "0">:$padHigh);
  let assemblyFormat = [{$input $filter $output attr-dict `:`
    type($input) type($filter) type($output)}];
}

def TileMegaConv2dOp : Tile_Op<"mega_conv2d"> {
  let summary = "NHWC INT8 convolution stage inside a Tile MegaKernel";
  let arguments = (ins MemRefRankOf<[I8], [4]>:$input,
                       MemRefRankOf<[I8], [4]>:$filter,
                       MemRefRankOf<[I32], [1]>:$bias,
                       MemRefRankOf<[F32], [1]>:$scale,
                       MemRefRankOf<[I8], [1]>:$lut,
                       MemRefRankOf<[I8, F32], [4]>:$output,
                       I64Attr:$stride,
                       I64Attr:$padLow,
                       I64Attr:$padHigh,
                       I64Attr:$activation,
                       F32Attr:$outputScale);
  let assemblyFormat = [{$input $filter $bias $scale $lut $output attr-dict `:`
    type($input) type($filter) type($bias) type($scale) type($lut) type($output)}];
}

def TileMegaConv2dDepthwiseOp : Tile_Op<"mega_conv2d_depthwise"> {
  let summary = "NHWC INT8 depthwise stage inside a Tile MegaKernel";
  let arguments = (ins MemRefRankOf<[I8], [4]>:$input,
                       MemRefRankOf<[I8], [4]>:$filter,
                       MemRefRankOf<[I32], [1]>:$bias,
                       MemRefRankOf<[F32], [1]>:$scale,
                       MemRefRankOf<[I8], [1]>:$lut,
                       MemRefRankOf<[I8, F32], [4]>:$output,
                       I64Attr:$stride,
                       I64Attr:$padLow,
                       I64Attr:$padHigh,
                       I64Attr:$activation,
                       F32Attr:$outputScale);
  let assemblyFormat = [{$input $filter $bias $scale $lut $output attr-dict `:`
    type($input) type($filter) type($bias) type($scale) type($lut) type($output)}];
}

def TileMegaMaxPool2dOp : Tile_Op<"mega_max_pool2d"> {
  let summary = "INT8 MaxPool2D stage inside a Tile MegaKernel";
  let arguments = (ins MemRefRankOf<[I8], [4]>:$input,
                       MemRefRankOf<[I8], [4]>:$output,
                       I64Attr:$kernel,
                       I64Attr:$stride,
                       I64Attr:$padding,
                       BoolAttr:$finalOutput);
  let assemblyFormat = [{$input $output attr-dict `:`
    type($input) type($output)}];
}

def TileMegaGlobalAvgPoolOp : Tile_Op<"mega_global_avg_pool"> {
  let summary = "Global-average an INT8 NHWC tensor inside a MegaKernel";
  let arguments = (ins MemRefRankOf<[I8], [4]>:$input,
                       MemRefRankOf<[I8], [4]>:$output,
                       F32Attr:$inputScale,
                       F32Attr:$outputScale);
  let assemblyFormat = [{$input $output attr-dict `:`
    type($input) type($output)}];
}

def TileMegaInt8MulOp : Tile_Op<"mega_int8_mul"> {
  let summary = "Scaled INT8 broadcast multiply inside a MegaKernel";
  let arguments = (ins MemRefRankOf<[I8], [4]>:$lhs,
                       MemRefRankOf<[I8], [4]>:$rhs,
                       MemRefRankOf<[I8], [4]>:$output,
                       F32Attr:$lhsScale,
                       F32Attr:$rhsScale,
                       F32Attr:$outputScale,
                       I64Attr:$activation);
  let assemblyFormat = [{$lhs $rhs $output attr-dict `:`
    type($lhs) type($rhs) type($output)}];
}

def TileMegaInt8AddOp : Tile_Op<"mega_int8_add"> {
  let summary = "Scaled INT8 residual add inside a MegaKernel";
  let arguments = (ins MemRefRankOf<[I8], [4]>:$lhs,
                       MemRefRankOf<[I8], [4]>:$rhs,
                       MemRefRankOf<[I8], [4]>:$output,
                       F32Attr:$lhsScale,
                       F32Attr:$rhsScale,
                       F32Attr:$outputScale,
                       I64Attr:$activation);
  let assemblyFormat = [{$lhs $rhs $output attr-dict `:`
    type($lhs) type($rhs) type($output)}];
}

def TileDepthwiseConv2dOp : Tile_Op<"tile_depthwise_conv2d"> {
  let summary = "Tiled depthwise 2D convolution (NHWC, multiplier=1)";
  let description = [{
    Input: [N, H, W, C], Filter: [KH, KW, C, 1], Output: [N, OH, OW, C].
    Lowered to channel-batched im2col + fat GEMM (N=lane), not GEMV.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [4]>:$input,
                       MemRefRankOf<[AnyType], [4]>:$filter,
                       MemRefRankOf<[AnyType], [4]>:$output,
                       DefaultValuedAttr<I64Attr, "0">:$padLow,
                       DefaultValuedAttr<I64Attr, "0">:$padHigh);
  let assemblyFormat = [{$input $filter $output attr-dict `:`
    type($input) type($filter) type($output)}];
}

#endif // TILE_OPS
