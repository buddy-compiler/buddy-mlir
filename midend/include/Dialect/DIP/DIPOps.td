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
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def DIP_ConstantPadding : I32EnumAttrCase<"ConstantPadding", 0, "CONSTANT_PADDING">;
def DIP_ReplicatePadding : I32EnumAttrCase<"ReplicatePadding", 1, "REPLICATE_PADDING">;

def DIP_NearestNeighbourInterpolation : I32EnumAttrCase<"NearestNeighbourInterpolation", 0,
                                        "NEAREST_NEIGHBOUR_INTERPOLATION">;
def DIP_BilinearInterpolation : I32EnumAttrCase<"BilinearInterpolation", 1,
                                "BILINEAR_INTERPOLATION">;

def DIP_BoundaryOption : I32EnumAttr<"BoundaryOption",
    "Specifies desired method of boundary extrapolation during image processing.",
    [
      DIP_ConstantPadding,
      DIP_ReplicatePadding
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::buddy::dip";
}

def DIP_InterpolationType : I32EnumAttr<"InterpolationType",
    "Specifies desired type of interpolation/extrapolation during image processing.",
    [
      DIP_NearestNeighbourInterpolation,
      DIP_BilinearInterpolation
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::buddy::dip";
}

def DIP_BoundaryOptionAttr : EnumAttr<DIP_Dialect, DIP_BoundaryOption, "boundary_option"> {
  let assemblyFormat = "`<` $value `>`";
}
def DIP_InterpolationAttr : EnumAttr<DIP_Dialect, DIP_InterpolationType, "interpolation_type">;

def DIP_Corr2DOp : DIP_Op<"corr_2d"> {
  let summary = [{This operation is used for performing 2D correlation on an image.
    The 2D correlation API provided by the linalg dialect is more suited for
    applications in which boundary extrapolation is not explicitly required.
    Due to this, dimensions of output are always less than the input dimensions after
    using linalg dialect's 2D correlation API.

    dip.corr_2d performs boundary extrapolation for making the size of the output image
    equal to the size of the input image. Boundary extrapolation can be done using
    different methods, supported options are:
      a. Constant Padding : Uses a constant for padding whole extra region in input image
      for obtaining the boundary extrapolated output image. (kkk|abcdefg|kkk)
      b. Replicate Padding : Uses last/first element of respective column/row for padding
      the extra region used for creating the boundary extrapolated output image. (aaa|abcdefg|ggg)
    For example:

    ```mlir
      dip.corr_2d CONSTANT_PADDING %inputImage, %kernel, %output, %centerX, %centerY, %constantValue
          : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index
    ```
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref",
                           [MemRead]>:$memrefI,
                       Arg<AnyRankedOrUnrankedMemRef, "kernelMemref",
                           [MemRead]>:$memrefK,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefCO,
                       Index : $centerX,
                       Index : $centerY,
                       AnyTypeOf<[AnyI8, AnyI32, AnyI64, AnyFloat]> : $constantValue,
                       DIP_BoundaryOptionAttr:$boundary_option);

  let assemblyFormat = [{
    $boundary_option $memrefI `,` $memrefK `,` $memrefCO `,` $centerX `,` $centerY `,` $constantValue attr-dict `:` type($memrefI) `,` type($memrefK) `,` type($memrefCO) `,` type($centerX) `,` type($centerY) `,` type($constantValue)
  }];
}

def DIP_CorrFFT2DOp : DIP_Op<"corrfft_2d">
{
  let summary = [{ abc def ghi jkl mno pqr stu vwx yz }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemrefReal",
                           [MemRead]>:$memrefIReal,
                       Arg<AnyRankedOrUnrankedMemRef, "inputMemrefImag",
                           [MemRead]>:$memrefIImag,
                       Arg<AnyRankedOrUnrankedMemRef, "kernelMemrefReal",
                           [MemRead]>:$memrefKReal,
                       Arg<AnyRankedOrUnrankedMemRef, "kernelMemrefImag",
                           [MemRead]>:$memrefKImag,
                       Arg<AnyRankedOrUnrankedMemRef, "intermediateMemrefReal",
                           [MemRead]>:$memrefIntReal,
                       Arg<AnyRankedOrUnrankedMemRef, "intermediateMemrefImag",
                           [MemRead]>:$memrefIntImag);

  let assemblyFormat = [{
    $memrefIReal `,` $memrefIImag `,` $memrefKReal `,` $memrefKImag `,` $memrefIntReal `,` 
    $memrefIntImag attr-dict `:` 
    type($memrefIReal) `,` type($memrefIImag) `,` type($memrefKReal) `,` 
    type($memrefKImag) `,` type($memrefIntReal) `,` type($memrefIntImag)
  }];
}

def DIP_Rotate2DOp : DIP_Op<"rotate_2d"> {
  let summary = [{This operation intends to provide utility for rotating images via the DIP dialect.
  Image rotation has many applications such as data augmentation, alignment adjustment, etc. and
  can thus be used in native MLIR pipelines involving above mentioned uses.

  Standard affine rotation matrix [cosθ   sinθ] is used whenever |tan(θ/2)| > 8.1 for the user defined θ.
				                          [-sinθ  cosθ]
  In all other cases, the 3 shear method is used to simplify calculations and minimize generated
  artifacts. The equivalent 3 shear matrix combination is as follows :
  [1  -tan(θ/2)]   *   [ 1       0]   *   [1    -tan(θ/2)]
  [0      1    ]       [sinθ     1]       [0        1    ]

  For example:

  ```mlir
  dip.rotate_2d %inputImage, %angle, %outputImage : memref<?x?xf32>, f32, memref<?x?xf32>
  ```
}];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref",
                           [MemRead]>:$memrefI,
                       F32 : $angle,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefO);

  let assemblyFormat = [{
    $memrefI `,` $angle `,` $memrefO attr-dict `:` type($memrefI) `,` type($angle) `,` type($memrefO)
  }];
}

def DIP_Resize2DOp : DIP_Op<"resize_2d">
{
  let summary = [{
    This operation intends to provide a utility for resizing images using the DIP dialect.
    Image resizing has many applications such as data augmentation, dimension adjustment in ML
    models, etc. and can thus be used in native MLIR pipelines catering to above mentioned
    use-cases.

    As of now, two different mechanisms for pixel interpolation are provided namely nearest
    neighbour interpolation and bilinear interpolation. The user can specify the desired type of
    interpolation via an attribute provided as argument to the operation. The operation also
    expects scaling ratios (Input image dimension / Output image dimension) for both dimensions
    of input and output images as arguments.

    The operation is flexible for its use with images of different sizes without necessarily
    lowering it every time for each new image (Refer to the example provided in examples
    directory for the DIP dialect).

    Syntax :

    ```mlir
    dip.resize_2d INTERPOLATION_TYPE %inputImage, %horizontal_scaling_factor, %vertical_scaling_factor, %outputImage : memref<?x?xf32>, f32, f32, memref<?x?xf32>
    ```

    where ```INTERPOLATION_TYPE``` can be ```NEAREST_NEIGHBOUR_INTERPOLATION``` or
    ```BILINEAR_INTERPOLATION```.
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref",
                           [MemRead]>:$memrefI,
                       F32 : $horizontal_scaling_factor,
                       F32 : $vertical_scaling_factor,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefO,
                       DIP_InterpolationAttr:$interpolation_type);

  let assemblyFormat = [{
     $interpolation_type $memrefI `,` $horizontal_scaling_factor `,` $vertical_scaling_factor `,` $memrefO attr-dict `:` type($memrefI) `,` type($horizontal_scaling_factor) `,` type($vertical_scaling_factor) `,` type($memrefO)
  }];
}

def DIP_Erosion2DOp : DIP_Op<"erosion_2d"> {
  let summary = [{This operation aims to provide utility to perform Erosion on
                      a 2d single channel image.}];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref",
                           [MemRead]>:$memrefI,
                       Arg<AnyRankedOrUnrankedMemRef, "kernelMemref",
                           [MemRead]>:$memrefK,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefCO,
                       Arg<AnyRankedOrUnrankedMemRef, "copyMemref",
                           [MemRead]>:$memrefCopy,
                       Index : $centerX,
                       Index : $centerY,
                       Index : $iterations,
                       AnyTypeOf<[AnyI8, AnyI32, AnyI64, AnyFloat]> : $constantValue,
                       DIP_BoundaryOptionAttr:$boundary_option);

  let assemblyFormat = [{
    $boundary_option $memrefI `,`  $memrefK `,` $memrefCO `,` $memrefCopy `,` $centerX `,` $centerY `,`$iterations `,` $constantValue  attr-dict `:` type($memrefI) `,` type($memrefK) `,` type($memrefCO) `,` type($memrefCopy) `,` type($centerX) `,` type($centerY) `,`type($iterations) `,` type($constantValue)
  }];
}

def DIP_Dilation2DOp : DIP_Op<"dilation_2d"> {
  let summary = [{This operation aims to provide utility to perform Dilation on
                      a 2d single channel image.}];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref",
                           [MemRead]>:$memrefI,
                       Arg<AnyRankedOrUnrankedMemRef, "kernelMemref",
                           [MemRead]>:$memrefK,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefCO,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefCopy,
                       Index : $centerX,
                       Index : $centerY,
                       Index : $iterations,
                       AnyTypeOf<[AnyI8, AnyI32, AnyI64, AnyFloat]> : $constantValue,
                       DIP_BoundaryOptionAttr:$boundary_option);

  let assemblyFormat = [{
    $boundary_option $memrefI `,` $memrefK `,` $memrefCO `,` $memrefCopy `,` $centerX `,` $centerY `,` $iterations `,` $constantValue  attr-dict `:` type($memrefI) `,` type($memrefK) `,` type($memrefCO) `,` type($memrefCopy) `,` type($centerX) `,` type($centerY) `,` type($iterations) `,` type($constantValue)
  }];
}

def DIP_Opening2DOp : DIP_Op<"opening_2d"> {
  let summary = [{This operation aims to provide utility to perform Opening on
                      a 2d single channel image.}];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref",
                           [MemRead]>:$memrefI,
                       Arg<AnyRankedOrUnrankedMemRef, "kernelMemref",
                           [MemRead]>:$memrefK,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefCO,
                       Arg<AnyRankedOrUnrankedMemRef, "outputinterMemref",
                           [MemRead]>:$memrefCOI,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemrefcopy",
                           [MemRead]>:$memrefCopy,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefCopyone,
                       Index : $centerX,
                       Index : $centerY,
                       Index : $iterations,
                       AnyTypeOf<[AnyI8, AnyI32, AnyI64, AnyFloat]> : $constantValue,
                       DIP_BoundaryOptionAttr:$boundary_option);

  let assemblyFormat = [{
    $boundary_option $memrefI `,` $memrefK `,` $memrefCO `,` $memrefCOI `,` $memrefCopy `,` $memrefCopyone `,` $centerX `,` $centerY `,` $iterations `,` $constantValue  attr-dict `:` type($memrefI) `,` type($memrefK) `,` type($memrefCO) `,` type($memrefCOI) `,` type($memrefCopy) `,` type($memrefCopyone) `,` type($centerX) `,` type($centerY) `,` type($iterations) `,` type($constantValue)
  }];
}

def DIP_Closing2DOp : DIP_Op<"closing_2d"> {
  let summary = [{This operation aims to provide utility to perform Closing transformation on
                      a 2d single channel image.}];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref",
                           [MemRead]>:$memrefI,
                       Arg<AnyRankedOrUnrankedMemRef, "kernelMemref",
                           [MemRead]>:$memrefK,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefCO,
                       Arg<AnyRankedOrUnrankedMemRef, "outputinterMemref",
                           [MemRead]>:$memrefCOI,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemrefcopy",
                           [MemRead]>:$memrefCopy,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefCopyone,
                       Index : $centerX,
                       Index : $centerY,
                       Index : $iterations,
                       AnyTypeOf<[AnyI8, AnyI32, AnyI64, AnyFloat]> : $constantValue,
                       DIP_BoundaryOptionAttr:$boundary_option);

  let assemblyFormat = [{
    $boundary_option $memrefI `,`  $memrefK `,` $memrefCO `,` $memrefCOI `,` $memrefCopy `,` $memrefCopyone `,` $centerX `,` $centerY `,` $iterations `,` $constantValue  attr-dict `:` type($memrefI) `,` type($memrefK) `,` type($memrefCO) `,` type($memrefCOI) `,` type($memrefCopy) `,` type($memrefCopyone) `,` type($centerX) `,` type($centerY) `,` type($iterations) `,` type($constantValue)
  }];
}

def DIP_TopHat2DOp : DIP_Op<"tophat_2d"> {
  let summary = [{This operation aims to provide utility to perform tophat operation on
                      a 2d single channel image.}];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref",
                           [MemRead]>:$memrefI,
                       Arg<AnyRankedOrUnrankedMemRef, "kernelMemref",
                           [MemRead]>:$memrefK,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefCO,
                       Arg<AnyRankedOrUnrankedMemRef, "outputinterMemref",
                           [MemRead]>:$memrefCOI,
                       Arg<AnyRankedOrUnrankedMemRef, "outputinterrMemref" ,
                           [MemRead]>:$memrefCOII,
                       Arg<AnyRankedOrUnrankedMemRef, "inputinterMemref",
                           [MemRead]>:$memrefIR,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemrefcopy",
                           [MemRead]>:$memrefCopy,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefCopyone,
                       Index : $centerX,
                       Index : $centerY,
                       Index : $iterations,
                       AnyTypeOf<[AnyI8, AnyI32, AnyI64, AnyFloat]> : $constantValue,
                       DIP_BoundaryOptionAttr:$boundary_option);

  let assemblyFormat = [{
    $boundary_option $memrefI  `,`  $memrefK `,` $memrefCO `,` $memrefCOI `,` $memrefCOII `,` $memrefIR `,` $memrefCopy `,` $memrefCopyone `,` $centerX `,` $centerY `,` $iterations `,` $constantValue   attr-dict `:` type($memrefI) `,` type($memrefK)  `,` type($memrefCO) `,` type($memrefCOI)   `,` type($memrefCOII) `,` type($memrefIR) `,` type($memrefCopy) `,` type($memrefCopyone) `,` type($centerX) `,` type($centerY) `,` type($iterations) `,` type($constantValue)
  }];
}

def DIP_BottomHat2DOp : DIP_Op<"bottomhat_2d"> {
  let summary = [{This operation aims to provide utility to perform Bottomhat operation on
                      a 2d single channel image.}];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref",
                           [MemRead]>:$memrefI,
                       Arg<AnyRankedOrUnrankedMemRef, "kernelMemref",
                           [MemRead]>:$memrefK,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefCO,
                       Arg<AnyRankedOrUnrankedMemRef, "outputinterMemref",
                           [MemRead]>:$memrefCOI,
                       Arg<AnyRankedOrUnrankedMemRef, "outputinterrMemref" ,
                           [MemRead]>:$memrefCOII,
                       Arg<AnyRankedOrUnrankedMemRef, "inputMemrefinter",
                           [MemRead]>:$memrefIR,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemrefcopy",
                           [MemRead]>:$memrefCopy,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefCopyone,
                       Index : $centerX,
                       Index : $centerY,
                       Index : $iterations,
                       AnyTypeOf<[AnyI8, AnyI32, AnyI64, AnyFloat]> : $constantValue,
                       DIP_BoundaryOptionAttr:$boundary_option);

  let assemblyFormat = [{
    $boundary_option $memrefI  `,` $memrefK `,` $memrefCO `,` $memrefCOI `,` $memrefCOII `,` $memrefIR `,` $memrefCopy `,` $memrefCopyone `,` $centerX `,` $centerY `,` $iterations `,` $constantValue  attr-dict `:` type($memrefI) `,` type($memrefK) `,` type($memrefCO) `,` type($memrefCOI) `,` type($memrefCOII) `,`type($memrefIR) `,` type($memrefCopy) `,` type($memrefCopyone) `,` type($centerX) `,` type($centerY) `,`type($iterations) `,` type($constantValue)
  }];
}

def DIP_MorphGrad2DOp : DIP_Op<"morphgrad_2d"> {
  let summary = [{This operation aims to provide utility to perform Morphological Gradient transformation on
                      a 2d single channel image.}];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref",
                           [MemRead]>:$memrefI,
                       Arg<AnyRankedOrUnrankedMemRef, "kernelMemref",
                           [MemRead]>:$memrefK,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefCO,
                       Arg<AnyRankedOrUnrankedMemRef, "outputinterMemref",
                           [MemRead]>:$memrefCOI,
                       Arg<AnyRankedOrUnrankedMemRef, "outputinterMemref1",
                           [MemRead]>:$memrefCOII,
                       Arg<AnyRankedOrUnrankedMemRef, "inputMemrefinter",
                           [MemRead]>:$memrefIR,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemrefcopy",
                           [MemRead]>:$memrefCopy,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefCopyone,
                       Index : $centerX,
                       Index : $centerY,
                       Index : $iterations,
                       AnyTypeOf<[AnyI8, AnyI32, AnyI64, AnyFloat]> : $constantValue,
                       DIP_BoundaryOptionAttr:$boundary_option);

  let assemblyFormat = [{
    $boundary_option $memrefI `,`  $memrefK `,` $memrefCO `,` $memrefCOI `,` $memrefCOII `,` $memrefIR `,` $memrefCopy `,` $memrefCopyone `,` $centerX `,` $centerY `,` $iterations `,` $constantValue  attr-dict `:` type($memrefI) `,` type($memrefK) `,` type($memrefCO) `,` type($memrefCOI) `,` type($memrefCOII) `,` type($memrefIR) `,` type($memrefCopy) `,` type($memrefCopyone) `,` type($centerX) `,` type($centerY) `,` type($iterations) `,` type($constantValue)
  }];
}

#endif // DIP_DIPOPS_TD
