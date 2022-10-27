//===- DAPOps.td - dap Dialect Ops -------------------------*- tablegen -*-===//
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
// This is the top level file for operations in the DAP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DAP_DAPOPS_TD
#define DAP_DAPOPS_TD

include "DAPDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def DAP_FirOp : DAP_Op<"fir"> {
  let summary = [{FIR filter, a finite impulse response (FIR) filter is a linear
  time-invariant filter that is used to filter a signal. It is a linear
  convolution of the input signal with a filter kernel.

  ```mlir
    dsp.fir %input, %kernel, %output :memref<?x?xf32>, memref<?x?xf32>, 
            memref<?x?xf32>
  ```
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref",
                           [MemRead]>:$memrefI,
                       Arg<AnyRankedOrUnrankedMemRef, "kernelMemref",
                           [MemRead]>:$memrefK,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefO);

  let assemblyFormat = [{
    $memrefI `,` $memrefK `,` $memrefO attr-dict `:` type($memrefI) `,` type($memrefK) `,` type($memrefO) 
  }];
}

def DAP_BiquadOp : DAP_Op<"biquad">
{
  let summary = [{Biquad filter, a infinite impulse response (IIR) filter.

  ```mlir
    dap.biquad %input, %kernel, %output :memref<?x?xf32>, memref<?x?xf32>, 
            memref<?x?xf32>
  ```
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref",
                           [MemRead]>:$memrefI,
                       Arg<AnyRankedOrUnrankedMemRef, "kernelMemref",
                           [MemRead]>:$memrefK,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefO);

  let assemblyFormat = [{
    $memrefI `,` $memrefK `,` $memrefO attr-dict `:` type($memrefI) `,` type($memrefK) `,` type($memrefO) 
  }];
}

def DAP_IirOp : DAP_Op<"iir"> {
  let summary = [{IIR filter, a infinite impulse response (IIR), Unlike FIR filters, 
  they have a feedback(a recursive part of a filter).

  ```mlir
    dsp.iir %input, %kernel, %output :memref<?xf32>, memref<?x?xf32>, 
            memref<?xf32>
  ```
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref",
                           [MemRead]>:$memrefI,
                       Arg<AnyRankedOrUnrankedMemRef, "kernelMemref",
                           [MemRead]>:$memrefK,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefO);

  let assemblyFormat = [{
    $memrefI `,` $memrefK `,` $memrefO attr-dict `:` type($memrefI) `,` type($memrefK) `,` type($memrefO) 
  }];
}

#endif // DAP_DAPOPS_TD
