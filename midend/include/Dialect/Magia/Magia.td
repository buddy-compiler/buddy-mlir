//===-- Magia.td - MAGIA dialect operation definitions ------- tablegen --===//
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
// This file defines the operations of the MAGIA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MAGIA_OPS
#define MAGIA_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// MAGIA dialect definition
//===----------------------------------------------------------------------===//

def Magia_Dialect : Dialect {
  let name = "magia";
  let cppNamespace = "::buddy::magia";
  let summary = "Dialect to target the MAGIA mesh of RedMulE tiles";
  let description = [{
    MAGIA is a mesh of tiles, each a RISC-V core with a RedMulE array for
    FP16 matrix products, a DMA and a megabyte of memory. A host hart runs the
    program and hands the products to the mesh. For more details about MAGIA,
    please see the [MAGIA repo](https://github.com/pulp-platform/MAGIA).
  }];
}

//===----------------------------------------------------------------------===//
// MAGIA operation definitions
//===----------------------------------------------------------------------===//

class Magia_Op<string mnemonic, list<Trait> traits = []> :
  Op<Magia_Dialect, mnemonic, traits> {}

def GemmOp : Magia_Op<"gemm"> {
  let summary = "General matrix product on the mesh";
  let description = [{
    Z = Z + X * W in FP16, as RedMulE computes it: X is M x N, W is N x K and
    Z, which holds Y going in, is M x K. The tiles share the columns of W.

    Example:

    ```mlir
    magia.gemm %x %w %z : memref<32x768xf16> memref<768x768xf16>
                          memref<32x768xf16>
    ```
  }];
  let arguments = (ins Arg<MemRefRankOf<[F16], [2]>, "", [MemRead]>:$x,
                       Arg<MemRefRankOf<[F16], [2]>, "", [MemRead]>:$w,
                       Arg<MemRefRankOf<[F16], [2]>, "",
                           [MemRead, MemWrite]>:$z);
  let assemblyFormat = "$x $w $z attr-dict `:` type($x) type($w) type($z)";
}

#endif // MAGIA_OPS
