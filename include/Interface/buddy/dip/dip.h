//===- dip.h --------------------------------------------------------===//
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

#include <Interface/buddy/dip/memref.h>

namespace dip {
namespace detail {
// Functions present inside dip::detail are not meant to be called by users
// directly.
// Declare the Corr2D C interface.
extern "C" {
void _mlir_ciface_corr_2d_constant_padding(
    MemRef_descriptor input, MemRef_descriptor kernel, MemRef_descriptor output,
    unsigned int centerX, unsigned int centerY, float constantValue);

void _mlir_ciface_corr_2d_replicate_padding(
    MemRef_descriptor input, MemRef_descriptor kernel, MemRef_descriptor output,
    unsigned int centerX, unsigned int centerY, float constantValue);

void _mlir_ciface_rotate_2d(MemRef_descriptor input, float angleValue,
                            MemRef_descriptor output);
}
} // namespace detail

enum class BOUNDARY_OPTION { CONSTANT_PADDING, REPLICATE_PADDING };
enum class ANGLE_TYPE { DEGREE, RADIAN };

void Corr2D(MemRef_descriptor input, MemRef_descriptor kernel,
            MemRef_descriptor output, unsigned int centerX,
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

MemRef_descriptor Rotate2D(MemRef_descriptor input, float angle,
                           ANGLE_TYPE angleType) {
  float angleRad;

  if (angleType == ANGLE_TYPE::DEGREE)
    angleRad = M_PI * angle / 180;
  else
    angleRad = angle;

  float sinAngle = std::sin(angleRad);
  float cosAngle = std::cos(angleRad);

  int outputRows = std::round(std::abs(input->sizes[0] * cosAngle) +
                              std::abs(input->sizes[1] * sinAngle)) + 1;
  int outputCols = std::round(std::abs(input->sizes[1] * cosAngle) +
                              std::abs(input->sizes[0] * sinAngle)) + 1;
  float *outputAlign = (float *)malloc(outputRows * outputCols * sizeof(float));

  for (int i = 0; i < outputRows; ++i)
    for (int j = 0; j < outputCols; ++j)
      outputAlign[i * outputRows + j] = 0;

  float *allocated = (float *)malloc(1 * sizeof(float));
  intptr_t sizesOutput[2] = {outputRows, outputCols};
  intptr_t stridesOutput[2] = {outputRows, outputCols};
  MemRef_descriptor output =
      MemRef_Descriptor(allocated, outputAlign, 0, sizesOutput, stridesOutput);

  detail::_mlir_ciface_rotate_2d(input, angleRad, output);

  return output;
}
} // namespace dip

#endif
