//===- DIPOps.td - dip Dialect Ops -------------------------*- tablegen -*-===//
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
// This is the top level file for operations in the DIP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DIP_DIPOPS_TD
#define DIP_DIPOPS_TD

include "DIPDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def DIP_ConstantPadding : StrEnumAttrCase<"ConstantPadding", -1, "CONSTANT_PADDING">;
def DIP_ReplicatePadding : StrEnumAttrCase<"ReplicatePadding", -1, "REPLICATE_PADDING">;

def DIP_BoundaryOptionAttr : StrEnumAttr<"BoundaryOptionAttr",
    "Specifies desired method of boundary extrapolation during image processing.",
    [
      DIP_ConstantPadding,
      DIP_ReplicatePadding
    ]>{
  let cppNamespace = "::buddy::dip";
}

def DIP_Corr2DOp : DIP_Op<"corr_2d">
{
  let summary = [{This operation is used for performing 2D correlation on an image.
  The 2D correlation API provided by the linalg dialect is more suited for 
  applications in which boundary extrapolation is not explicitly required.
  Due to this, dimensions of output are always less than the input dimensions after
  using linalg dialect's 2D correlation API.

  dip.corr_2d performs boundary extrapolation for making the size of the output image
  equal to the size of the input image. Boundary extrapolation can be done using
  different methods, supported options are : 
    a. Constant Padding : Uses a constant for padding whole extra region in input image
    for obtaining the boundary extrapolated output image. (kkk|abcdefg|kkk)
    b. Replicate Padding : Uses last/first element of respective column/row for padding
    the extra region used for creating the boundary extrapolated output image. (aaa|abcdefg|ggg)
  For example: 

  ```mlir
    dip.corr_2d %inputImage, %kernel, %output, %centerX, %centerY, %constantValue
                {boundary_option = "CONSTANT_PADDING"} : memref<?x?xf32>, memref<?x?xf32>,
                memref<?x?xf32>, index, index, index
  ```
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref",
                           [MemRead]>:$memrefI,
                       Arg<AnyRankedOrUnrankedMemRef, "kernelMemref",
                           [MemRead]>:$memrefK,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefCO,
                       Index : $centerX, Index : $centerY, F32 : $constantValue,
                       DIP_BoundaryOptionAttr:$boundary_option);

  let assemblyFormat = [{
    $memrefI `,` $memrefK `,` $memrefCO `,` $centerX `,` $centerY `,` $constantValue attr-dict `:` type($memrefI) `,` type($memrefK) `,` type($memrefCO) `,` type($centerX) `,` type($centerY) `,` type($constantValue)
  }];
}

#endif // DIP_DIPOPS_TD
