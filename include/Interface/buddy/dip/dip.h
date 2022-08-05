//===- dip.h --------------------------------------------------------------===//
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
// Header file for DIP dialect specific operations and other entities.
//
//===----------------------------------------------------------------------===//

#ifndef INCLUDE_DIP
#define INCLUDE_DIP

#include "Interface/buddy/core/ImageContainer.h"

namespace dip {
enum class BOUNDARY_OPTION { CONSTANT_PADDING, REPLICATE_PADDING };
enum class ANGLE_TYPE { DEGREE, RADIAN };
enum class INTERPOLATION_TYPE {
  NEAREST_NEIGHBOUR_INTERPOLATION,
  BILINEAR_INTERPOLATION
};
enum class STRUCTURING_TYPE {FLAT, NONFLAT};

namespace detail {
// Functions present inside dip::detail are not meant to be called by users
// directly.
// Declare the Corr2D C interface.
extern "C" {
void _mlir_ciface_corr_2d_constant_padding(
    Img<float, 2> *input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    unsigned int centerX, unsigned int centerY, float constantValue);

void _mlir_ciface_corr_2d_replicate_padding(
    Img<float, 2> *input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    unsigned int centerX, unsigned int centerY, float constantValue);

void _mlir_ciface_rotate_2d(Img<float, 2> *input, float angleValue,
                            MemRef<float, 2> *output);

void _mlir_ciface_resize_2d_nearest_neighbour_interpolation(
    Img<float, 2> *input, float horizontalScalingFactor,
    float verticalScalingFactor, MemRef<float, 2> *output);

void _mlir_ciface_resize_2d_bilinear_interpolation(
    Img<float, 2> *input, float horizontalScalingFactor,
    float verticalScalingFactor, MemRef<float, 2> *output);

void _mlir_ciface_erosion_2d_constant_padding_flat(
    Img<float,2> *input, MemRef<float, 2> *kernel, MemRef<float, 2> *output, 
    unsigned int centerX, unsigned int centerY, float constantValue);   

void _mlir_ciface_erosion_2d_replicate_padding_flat(
    Img<float, 2> *input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    unsigned int centerX, unsigned int centerY, float constantValue);

void _mlir_ciface_erosion_2d_constant_padding_non_flat(
  Img<float, 2> *input, MemRef<float, 2> *kernel, MemRef<float, 2> *output, 
    unsigned int centerX, unsigned int centerY, float constantValue); 

void _mlir_ciface_erosion_2d_replicate_padding_non_flat(
  Img<float, 2> *input, MemRef<float, 2> *kernel, MemRef<float, 2> *output, 
    unsigned int centerX, unsigned int centerY, float constantValue);  

    void _mlir_ciface_dilation_2d_constant_padding_flat(
    Img<float,2> *input, MemRef<float, 2> *kernel, MemRef<float, 2> *output, 
    unsigned int centerX, unsigned int centerY, float constantValue);   

void _mlir_ciface_dilation_2d_replicate_padding_flat(
    Img<float, 2> *input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    unsigned int centerX, unsigned int centerY, float constantValue);

void _mlir_ciface_dilation_2d_constant_padding_non_flat(
  Img<float, 2> *input, MemRef<float, 2> *kernel, MemRef<float, 2> *output, 
    unsigned int centerX, unsigned int centerY, float constantValue); 

void _mlir_ciface_dilation_2d_replicate_padding_non_flat(
  Img<float, 2> *input, MemRef<float, 2> *kernel, MemRef<float, 2> *output, 
    unsigned int centerX, unsigned int centerY, float constantValue);  

}

MemRef<float, 2> Resize2D_Impl(Img<float, 2> *input, INTERPOLATION_TYPE type,
                               std::vector<float> scalingRatios,
                               intptr_t outputSize[2]) {
  MemRef<float, 2> output(outputSize);

  if (type == INTERPOLATION_TYPE::NEAREST_NEIGHBOUR_INTERPOLATION) {
    detail::_mlir_ciface_resize_2d_nearest_neighbour_interpolation(
        input, scalingRatios[0], scalingRatios[1], &output);
  } else if (type == INTERPOLATION_TYPE::BILINEAR_INTERPOLATION) {
    detail::_mlir_ciface_resize_2d_bilinear_interpolation(
        input, scalingRatios[0], scalingRatios[1], &output);
  } else {
    throw std::invalid_argument(
        "Please chose a supported type of interpolation "
        "(Nearest neighbour interpolation or Bilinear interpolation)\n");
  }

  return output;
}
} // namespace detail

void Corr2D(Img<float, 2> *input, MemRef<float, 2> *kernel,
            MemRef<float, 2> *output, unsigned int centerX,
            unsigned int centerY, BOUNDARY_OPTION option,
            float constantValue = 0) {
  if (option == BOUNDARY_OPTION::CONSTANT_PADDING) {
    detail::_mlir_ciface_corr_2d_constant_padding(
        input, kernel, output, centerX, centerY, constantValue);
  } else if (option == BOUNDARY_OPTION::REPLICATE_PADDING) {
    detail::_mlir_ciface_corr_2d_replicate_padding(input, kernel, output,
                                                   centerX, centerY, 0);
  }
}

MemRef<float, 2> Rotate2D(Img<float, 2> *input, float angle,
                          ANGLE_TYPE angleType) {
  float angleRad;

  if (angleType == ANGLE_TYPE::DEGREE)
    angleRad = M_PI * angle / 180;
  else
    angleRad = angle;

  float sinAngle = std::sin(angleRad);
  float cosAngle = std::cos(angleRad);

  int outputRows = std::round(std::abs(input->getSizes()[0] * cosAngle) +
                              std::abs(input->getSizes()[1] * sinAngle)) +
                   1;
  int outputCols = std::round(std::abs(input->getSizes()[1] * cosAngle) +
                              std::abs(input->getSizes()[0] * sinAngle)) +
                   1;

  intptr_t sizesOutput[2] = {outputRows, outputCols};
  MemRef<float, 2> output(sizesOutput);

  detail::_mlir_ciface_rotate_2d(input, angleRad, &output);

  return output;
}

MemRef<float, 2> Resize2D(Img<float, 2> *input, INTERPOLATION_TYPE type,
                          std::vector<float> scalingRatios) {
  if (!scalingRatios[0] || !scalingRatios[1]) {
    throw std::invalid_argument(
        "Please enter non-zero values of scaling ratios.\n"
        "Note : scaling ratio = "
        "input_image_dimension / output_image_dimension\n");
  }

  intptr_t outputSize[2] = {
      static_cast<unsigned int>(input->getSizes()[0] / scalingRatios[1]),
      static_cast<unsigned int>(input->getSizes()[1] / scalingRatios[0])};

  return detail::Resize2D_Impl(input, type, scalingRatios, outputSize);
}

MemRef<float, 2> Resize2D(Img<float, 2> *input, INTERPOLATION_TYPE type,
                          intptr_t outputSize[2]) {
  if (!outputSize[0] || !outputSize[1]) {
    throw std::invalid_argument(
        "Please enter non-zero values of output dimensions.\n");
  }

  std::vector<float> scalingRatios(2);
  scalingRatios[1] = input->getSizes()[0] * 1.0f / outputSize[0];
  scalingRatios[0] = input->getSizes()[1] * 1.0f / outputSize[1];

  return detail::Resize2D_Impl(input, type, scalingRatios, outputSize);
}
void Erosion2D(Img<float, 2> *input, MemRef<float, 2> *kernel, MemRef<float, 2> *output, unsigned int centerX, unsigned int centerY, BOUNDARY_OPTION option,STRUCTURING_TYPE type, float constantValue = 0)
{
   if(option == BOUNDARY_OPTION::CONSTANT_PADDING && type == STRUCTURING_TYPE::FLAT )
   {
   detail::_mlir_ciface_erosion_2d_constant_padding_flat(input, kernel, output, centerX, centerY, constantValue);
   }
   else if (option == BOUNDARY_OPTION::REPLICATE_PADDING && type == STRUCTURING_TYPE::FLAT){
    detail::_mlir_ciface_erosion_2d_replicate_padding_non_flat(input, kernel, output, centerX, centerY, 0);
   }
    else if (option == BOUNDARY_OPTION::REPLICATE_PADDING && type == STRUCTURING_TYPE::NONFLAT){
   detail::_mlir_ciface_erosion_2d_replicate_padding_non_flat(input, kernel, output, centerX, centerY, 0);
   }
 else if (option == BOUNDARY_OPTION::CONSTANT_PADDING && type == STRUCTURING_TYPE::NONFLAT){
  detail::_mlir_ciface_erosion_2d_constant_padding_non_flat(input, kernel, output, centerX, centerY, constantValue);
   }
}

void Dilation2D(Img<float, 2> *input, MemRef<float, 2> *kernel, MemRef<float, 2> *output, unsigned int centerX, unsigned int centerY, BOUNDARY_OPTION option, STRUCTURING_TYPE type, float constantValue = 0)
{
  if(option == BOUNDARY_OPTION::CONSTANT_PADDING && type == STRUCTURING_TYPE::FLAT )
   {
   detail::_mlir_ciface_dilation_2d_constant_padding_flat(input, kernel, output, centerX, centerY, constantValue);
   }
   else if (option == BOUNDARY_OPTION::REPLICATE_PADDING && type == STRUCTURING_TYPE::FLAT){
    detail::_mlir_ciface_dilation_2d_replicate_padding_non_flat(input, kernel, output, centerX, centerY, 0);
   }
    else if (option == BOUNDARY_OPTION::REPLICATE_PADDING && type == STRUCTURING_TYPE::NONFLAT){
   detail::_mlir_ciface_dilation_2d_replicate_padding_non_flat(input, kernel, output, centerX, centerY, 0);
   }
 else if (option == BOUNDARY_OPTION::CONSTANT_PADDING && type == STRUCTURING_TYPE::NONFLAT){
  detail::_mlir_ciface_dilation_2d_constant_padding_non_flat(input, kernel, output, centerX, centerY, constantValue);
   }
}

} // namespace dip
#endif
