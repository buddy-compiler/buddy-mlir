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

def DAP_MonoFormat : StrEnumAttrCase<"MonoFormat", -1, "MONO_FORMAT">;
def DAP_StereoFormat : StrEnumAttrCase<"StereoFormat", -1, "STEREO_FORMAT">;

def DAP_FormatOptionAttr : StrEnumAttr<"FormatOptionAttr",
    "Specifies audio format during audio processing.",
    [
      DAP_MonoFormat,
      DAP_StereoFormat
    ]>{
  let cppNamespace = "::buddy::dap";
}

def DAP_Fir : DAP_Op<"fir">
{
  let summary = [{FIR filter, a finite impulse response (FIR) filter is a linear
  time-invariant filter that is used to filter a signal. It is a linear
  convolution of the input signal with a filter kernel.

  ```mlir
    dsp.fir %input, %coefficient, %output :memref<?x?xf32>, memref<?x?xf32>, 
            memref<?x?xf32>
  ```
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref",
                           [MemRead]>:$memrefI,
                       Arg<AnyRankedOrUnrankedMemRef, "coefficientMemref",
                           [MemRead]>:$memrefC,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefO,
                       DIP_BoundaryOptionAttr:$boundary_option);

  let assemblyFormat = [{
    $memrefI `,` $memrefC `,` $memrefO `,` attr-dict `:` type($memrefI) `,` type($memrefC) `,` type($memrefO) 
  }];
}

#endif // DAP_DAPOPS_TD
     