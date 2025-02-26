//===- DIP.mlir -----------------------------------------------------------===//
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
// This file provides DIP dialect functions.
//
//===----------------------------------------------------------------------===//

func.func @corr_2d_constant_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %centerX : index, %centerY : index, %constantValue : f32) attributes{llvm.emit_c_interface}
{
  dip.corr_2d <CONSTANT_PADDING> %inputImage, %kernel, %outputImage, %centerX, %centerY, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  return
}

func.func @corr_2d_replicate_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %centerX : index, %centerY : index, %constantValue : f32) attributes{llvm.emit_c_interface}
{
  dip.corr_2d <REPLICATE_PADDING> %inputImage, %kernel, %outputImage, %centerX, %centerY , %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  return
}

func.func @corrfft_2d(%inputImageReal : memref<?x?xf32>, %inputImageImag : memref<?x?xf32>, %kernelReal : memref<?x?xf32>, %kernelImag : memref<?x?xf32>, %intermediateReal : memref<?x?xf32>, %intermediateImag : memref<?x?xf32>) attributes{llvm.emit_c_interface}
{
  dip.corrfft_2d %inputImageReal, %inputImageImag, %kernelReal, %kernelImag, %intermediateReal, %intermediateImag : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
  return
}

func.func @rotate_2d(%inputImage : memref<?x?xf32>, %angle : f32, %outputImage : memref<?x?xf32>) attributes{llvm.emit_c_interface}
{
  dip.rotate_2d %inputImage, %angle, %outputImage : memref<?x?xf32>, f32, memref<?x?xf32>
  return
}

func.func @rotate_4d_nhwc(%inputImage : memref<?x?x?x?xf32>, %angle : f32, %outputImage : memref<?x?x?x?xf32>) attributes{llvm.emit_c_interface}
{
  dip.rotate_4d NHWC %inputImage, %angle, %outputImage : memref<?x?x?x?xf32>, f32, memref<?x?x?x?xf32>
  return
}

func.func @rotate_4d_nchw(%inputImage : memref<?x?x?x?xf32>, %angle : f32, %outputImage : memref<?x?x?x?xf32>) attributes{llvm.emit_c_interface}
{
  dip.rotate_4d NCHW %inputImage, %angle, %outputImage : memref<?x?x?x?xf32>, f32, memref<?x?x?x?xf32>
  return
}

func.func @resize_2d_nearest_neighbour_interpolation(%inputImage : memref<?x?xf32>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %outputImage : memref<?x?xf32>) attributes{llvm.emit_c_interface}
{
  dip.resize_2d NEAREST_NEIGHBOUR_INTERPOLATION %inputImage, %horizontal_scaling_factor, %vertical_scaling_factor, %outputImage : memref<?x?xf32>, f32, f32, memref<?x?xf32>
  return
}

func.func @resize_2d_bilinear_interpolation(%inputImage : memref<?x?xf32>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %outputImage : memref<?x?xf32>) attributes{llvm.emit_c_interface}
{
  dip.resize_2d BILINEAR_INTERPOLATION %inputImage, %horizontal_scaling_factor, %vertical_scaling_factor, %outputImage : memref<?x?xf32>, f32, f32, memref<?x?xf32>
  return
}

func.func @resize_4d_nhwc_nearest_neighbour_interpolation(%inputImage : memref<?x?x?x?xf32>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %outputImage : memref<?x?x?x?xf32>) attributes{llvm.emit_c_interface}
{
  dip.resize_4d_nhwc NEAREST_NEIGHBOUR_INTERPOLATION %inputImage, %horizontal_scaling_factor, %vertical_scaling_factor, %outputImage : memref<?x?x?x?xf32>, f32, f32, memref<?x?x?x?xf32>
  return
}

func.func @resize_4d_nhwc_bilinear_interpolation(%inputImage : memref<?x?x?x?xf32>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %outputImage : memref<?x?x?x?xf32>) attributes{llvm.emit_c_interface}
{
  dip.resize_4d_nhwc BILINEAR_INTERPOLATION %inputImage, %horizontal_scaling_factor, %vertical_scaling_factor, %outputImage : memref<?x?x?x?xf32>, f32, f32, memref<?x?x?x?xf32>
  return
}

func.func @resize_4d_nchw_nearest_neighbour_interpolation(%inputImage : memref<?x?x?x?xf32>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %outputImage : memref<?x?x?x?xf32>) attributes{llvm.emit_c_interface}
{
  dip.resize_4d_nchw NEAREST_NEIGHBOUR_INTERPOLATION %inputImage, %horizontal_scaling_factor, %vertical_scaling_factor, %outputImage : memref<?x?x?x?xf32>, f32, f32, memref<?x?x?x?xf32>
  return
}

func.func @resize_4d_nchw_bilinear_interpolation(%inputImage : memref<?x?x?x?xf32>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %outputImage : memref<?x?x?x?xf32>) attributes{llvm.emit_c_interface}
{
  dip.resize_4d_nchw BILINEAR_INTERPOLATION %inputImage, %horizontal_scaling_factor, %vertical_scaling_factor, %outputImage : memref<?x?x?x?xf32>, f32, f32, memref<?x?x?x?xf32>
  return
}

func.func @erosion_2d_constant_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue: f32) attributes{llvm.emit_c_interface}
{
  dip.erosion_2d <CONSTANT_PADDING> %inputImage, %kernel, %outputImage, %copymemref, %centerX, %centerY, %iterations, %constantValue: memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @erosion_2d_replicate_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue : f32) attributes{llvm.emit_c_interface}
{
  dip.erosion_2d <REPLICATE_PADDING> %inputImage, %kernel, %outputImage, %copymemref, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @dilation_2d_constant_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue: f32) attributes{llvm.emit_c_interface}
{
  dip.dilation_2d <CONSTANT_PADDING> %inputImage,  %kernel, %outputImage, %copymemref, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @dilation_2d_replicate_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue : f32) attributes{llvm.emit_c_interface}
{
  dip.dilation_2d <REPLICATE_PADDING> %inputImage, %kernel, %outputImage, %copymemref, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @opening_2d_constant_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue: f32) attributes{llvm.emit_c_interface}
{
  dip.opening_2d <CONSTANT_PADDING> %inputImage, %kernel, %outputImage, %outputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @opening_2d_replicate_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue : f32) attributes{llvm.emit_c_interface}
{
  dip.opening_2d <REPLICATE_PADDING> %inputImage, %kernel, %outputImage, %outputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @closing_2d_constant_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue: f32) attributes{llvm.emit_c_interface}
{
  dip.closing_2d <CONSTANT_PADDING> %inputImage, %kernel, %outputImage, %outputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @closing_2d_replicate_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue : f32) attributes{llvm.emit_c_interface}
{
  dip.closing_2d <REPLICATE_PADDING> %inputImage,  %kernel, %outputImage, %outputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>,memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @tophat_2d_constant_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>,%outputImage2 : memref<?x?xf32>, %inputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue: f32) attributes{llvm.emit_c_interface}
{
  dip.tophat_2d <CONSTANT_PADDING> %inputImage, %kernel, %outputImage, %outputImage1, %outputImage2, %inputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @tophat_2d_replicate_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>, %outputImage2 : memref<?x?xf32>, %inputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue : f32) attributes{llvm.emit_c_interface}
{
  dip.tophat_2d <REPLICATE_PADDING> %inputImage, %kernel, %outputImage, %outputImage1, %outputImage2, %inputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @bottomhat_2d_constant_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>, %outputImage2 : memref<?x?xf32>, %inputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue: f32) attributes{llvm.emit_c_interface}
{
  dip.bottomhat_2d <CONSTANT_PADDING> %inputImage, %kernel, %outputImage, %outputImage1, %outputImage2, %inputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @bottomhat_2d_replicate_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>, %outputImage2 : memref<?x?xf32>, %inputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue : f32) attributes{llvm.emit_c_interface}
{
  dip.bottomhat_2d <REPLICATE_PADDING> %inputImage, %kernel, %outputImage, %outputImage1, %outputImage2, %inputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @morphgrad_2d_constant_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>,%outputImage2 : memref<?x?xf32>, %inputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index,%iterations : index, %constantValue: f32) attributes{llvm.emit_c_interface}
{
  dip.morphgrad_2d <CONSTANT_PADDING> %inputImage, %kernel, %outputImage, %outputImage1, %outputImage2,  %inputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index,index, f32
  return
}

func.func @morphgrad_2d_replicate_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>,%outputImage2 : memref<?x?xf32>, %inputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue : f32) attributes{llvm.emit_c_interface}
{
  dip.morphgrad_2d <REPLICATE_PADDING> %inputImage, %kernel, %outputImage, %outputImage1,%outputImage2, %inputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}
