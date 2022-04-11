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

def DAP_FirstStep : DAP_Op<"test">
{
  let summary = [{This operation is used for learning dialect operation desiging. 
  Just like printing "hello world" for learning programming.
  For example: 

  ```mlir
    dap.test %inputAudio, %output, %constantValue{format_option = "MONO_FORMAT"}:
                memref<?x?xf32>, memref<?x?xf32>, index
  ```
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref",
                           [MemRead]>:$memrefI,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefCO,
                       F32 : $constantValue,
                       DAP_FormatOptionAttr:$format_option);

  let assemblyFormat = [{
    $memrefI `,` $memrefCO `,` $constantValue attr-dict `:` type($memrefI) `,` type($memrefCO) `,` type($constantValue)
  }];
}

#endif // DAP_DAPOPS_TD
