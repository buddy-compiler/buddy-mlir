//===- DIP.h --------------------------------------------------------------===//
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

#ifndef FRONTEND_INTERFACES_BUDDY_DIP_DIP
#define FRONTEND_INTERFACES_BUDDY_DIP_DIP

#include "buddy/Core/Container.h"
#include "buddy/DIP/ImageContainer.h"
#include <math.h>
namespace dip {
// Availale types of boundary extrapolation techniques provided in DIP dialect.
enum class BOUNDARY_OPTION { CONSTANT_PADDING, REPLICATE_PADDING };

// Available ways of specifying angles in image processing operations provided
// by the DIP dialect.
enum class ANGLE_TYPE { DEGREE, RADIAN };

// Available ways of interpolation techniques in image processing operations
// provided by the DIP dialect.
enum class INTERPOLATION_TYPE {
  NEAREST_NEIGHBOUR_INTERPOLATION,
  BILINEAR_INTERPOLATION
};

namespace detail {
// Functions present inside dip::detail are not meant to be called by users
// directly.

extern "C" {
// Declare the Corr2D C interface.
void _mlir_ciface_corr_2d_constant_padding(
    Img<float, 2> *input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    unsigned int centerX, unsigned int centerY, float constantValue);

void _mlir_ciface_corr_2d_replicate_padding(
    Img<float, 2> *input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    unsigned int centerX, unsigned int centerY, float constantValue);

void _mlir_ciface_corrfft_2d(MemRef<float, 2> *inputReal,
                             MemRef<float, 2> *inputImag,
                             MemRef<float, 2> *kernelReal,
                             MemRef<float, 2> *kernelImag,
                             MemRef<float, 2> *intermediateReal,
                             MemRef<float, 2> *intermediateImag);

// Declare the Rotate2D C interface.
void _mlir_ciface_rotate_2d(Img<float, 2> *input, float angleValue,
                            MemRef<float, 2> *output);

// Declare the Resize2D C interface.
void _mlir_ciface_resize_2d_nearest_neighbour_interpolation(
    Img<float, 2> *input, float horizontalScalingFactor,
    float verticalScalingFactor, MemRef<float, 2> *output);

void _mlir_ciface_resize_2d_bilinear_interpolation(
    Img<float, 2> *input, float horizontalScalingFactor,
    float verticalScalingFactor, MemRef<float, 2> *output);

// Declare the Morphology 2D C interface.
void _mlir_ciface_erosion_2d_constant_padding(
    Img<float, 2> input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    MemRef<float, 2> *copymemref, unsigned int centerX, unsigned int centerY,
    unsigned int iterations, float constantValue);

void _mlir_ciface_erosion_2d_replicate_padding(
    Img<float, 2> input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    MemRef<float, 2> *copymemref, unsigned int centerX, unsigned int centerY,
    unsigned int iterations, float constantValue);

void _mlir_ciface_dilation_2d_constant_padding(
    Img<float, 2> input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    MemRef<float, 2> *copymemref, unsigned int centerX, unsigned int centerY,
    unsigned int iterations, float constantValue);

void _mlir_ciface_dilation_2d_replicate_padding(
    Img<float, 2> input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    MemRef<float, 2> *copymemref, unsigned int centerX, unsigned int centerY,
    unsigned int iterations, float constantValue);

void _mlir_ciface_opening_2d_constant_padding(
    Img<float, 2> input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    MemRef<float, 2> *output1, MemRef<float, 2> *copymemref,
    MemRef<float, 2> *copymemref1, unsigned int centerX, unsigned int centerY,
    unsigned int iterations, float constantValue);

void _mlir_ciface_opening_2d_replicate_padding(
    Img<float, 2> input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    MemRef<float, 2> *output1, MemRef<float, 2> *copymemref,
    MemRef<float, 2> *copymemref1, unsigned int centerX, unsigned int centerY,
    unsigned int iterations, float constantValue);

void _mlir_ciface_closing_2d_constant_padding(
    Img<float, 2> input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    MemRef<float, 2> *output1, MemRef<float, 2> *copymemref,
    MemRef<float, 2> *copymemref1, unsigned int centerX, unsigned int centerY,
    unsigned int iterations, float constantValue);

void _mlir_ciface_closing_2d_replicate_padding(
    Img<float, 2> input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    MemRef<float, 2> *output1, MemRef<float, 2> *copymemref,
    MemRef<float, 2> *copymemref1, unsigned int centerX, unsigned int centerY,
    unsigned int iterations, float constantValue);

void _mlir_ciface_tophat_2d_constant_padding(
    Img<float, 2> input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    MemRef<float, 2> *output1, MemRef<float, 2> *output2,
    MemRef<float, 2> *input1, MemRef<float, 2> *copymemref,
    MemRef<float, 2> *copymemref1, unsigned int centerX, unsigned int centerY,
    unsigned int iterations, float constantValue);

void _mlir_ciface_tophat_2d_replicate_padding(
    Img<float, 2> input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    MemRef<float, 2> *output1, MemRef<float, 2> *output2,
    MemRef<float, 2> *input1, MemRef<float, 2> *copymemref,
    MemRef<float, 2> *copymemref1, unsigned int centerX, unsigned int centerY,
    unsigned int iterations, float constantValue);

void _mlir_ciface_bottomhat_2d_constant_padding(
    Img<float, 2> input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    MemRef<float, 2> *output1, MemRef<float, 2> *output2,
    MemRef<float, 2> *input1, MemRef<float, 2> *copymemref,
    MemRef<float, 2> *copymemref1, unsigned int centerX, unsigned int centerY,
    unsigned int iterations, float constantValue);

void _mlir_ciface_bottomhat_2d_replicate_padding(
    Img<float, 2> input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    MemRef<float, 2> *output1, MemRef<float, 2> *output2,
    MemRef<float, 2> *input1, MemRef<float, 2> *copymemref,
    MemRef<float, 2> *copymemref1, unsigned int centerX, unsigned int centerY,
    unsigned int iterations, float constantValue);

void _mlir_ciface_morphgrad_2d_constant_padding(
    Img<float, 2> input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    MemRef<float, 2> *output1, MemRef<float, 2> *output2,
    MemRef<float, 2> *input1, MemRef<float, 2> *copymemref,
    MemRef<float, 2> *copymemref1, unsigned int centerX, unsigned int centerY,
    unsigned int iterations, float constantValue);

void _mlir_ciface_morphgrad_2d_replicate_padding(
    Img<float, 2> input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    MemRef<float, 2> *output1, MemRef<float, 2> *output2,
    MemRef<float, 2> *input1, MemRef<float, 2> *copymemref,
    MemRef<float, 2> *copymemref1, unsigned int centerX, unsigned int centerY,
    unsigned int iterations, float constantValue);
}

// Pad kernel as per the requirements for using FFT in convolution.
inline void padKernel(MemRef<float, 2> *kernel, unsigned int centerX,
                      unsigned int centerY, intptr_t *paddedSizes,
                      MemRef<float, 2> *kernelPaddedReal) {
  // Apply padding so that the center of kernel is at top left of 2D padded
  // container.
  for (long i = -static_cast<long>(centerY);
       i < static_cast<long>(kernel->getSizes()[0]) - centerY; ++i) {
    uint32_t r = (i < 0) ? (i + paddedSizes[0]) : i;
    for (long j = -static_cast<long>(centerX);
         j < static_cast<long>(kernel->getSizes()[1]) - centerX; ++j) {
      uint32_t c = (j < 0) ? (j + paddedSizes[1]) : j;
      kernelPaddedReal->getData()[r * paddedSizes[1] + c] =
          kernel
              ->getData()[(i + centerY) * kernel->getSizes()[1] + j + centerX];
    }
  }
}

// Helper function for applying 2D resize operation on images.
inline MemRef<float, 2> Resize2D_Impl(Img<float, 2> *input,
                                      INTERPOLATION_TYPE type,
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

// User interface for 2D Correlation.
inline void Corr2D(Img<float, 2> *input, MemRef<float, 2> *kernel,
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

inline void CorrFFT2D(Img<float, 2> *input, MemRef<float, 2> *kernel,
                      MemRef<float, 2> *output, unsigned int centerX,
                      unsigned int centerY, BOUNDARY_OPTION option,
                      float constantValue = 0) {
  // Calculate padding sizes.
  intptr_t paddedSizes[2] = {
      1 << static_cast<intptr_t>(
          ceil(log2(input->getSizes()[0] + kernel->getSizes()[0] - 1))),
      1 << static_cast<intptr_t>(
          ceil(log2(input->getSizes()[1] + kernel->getSizes()[1] - 1)))};
  intptr_t paddedTSizes[2] = {paddedSizes[1], paddedSizes[0]};

  // Declare padded containers for input image and kernel.
  // Also declare an intermediate container for calculation convenience.
  MemRef<float, 2> inputPaddedReal(paddedSizes);
  MemRef<float, 2> inputPaddedImag(paddedSizes);

  MemRef<float, 2> kernelPaddedReal(paddedSizes);
  MemRef<float, 2> kernelPaddedImag(paddedSizes);

  MemRef<float, 2> intermediateReal(paddedTSizes);
  MemRef<float, 2> intermediateImag(paddedTSizes);

  intptr_t flippedKernelSizeRows = kernel->getSizes()[0];
  intptr_t flippedKernelSizeCols = kernel->getSizes()[1];
  intptr_t flippedKernelSizes[2] = {flippedKernelSizeRows,
                                    flippedKernelSizeCols};
  MemRef<float, 2> flippedKernel(flippedKernelSizes);

  for (uint32_t i = 0; i < kernel->getSizes()[0]; ++i)
    for (uint32_t j = 0; j < kernel->getSizes()[1]; ++j) {
      flippedKernel.getData()[i * kernel->getSizes()[1] + j] =
          kernel->getData()[(kernel->getSizes()[0] - 1 - i) *
                                kernel->getSizes()[1] +
                            kernel->getSizes()[1] - 1 - j];
    }

  if (option == BOUNDARY_OPTION::CONSTANT_PADDING) {
    for (uint32_t i = 0; i < paddedSizes[0]; ++i) {
      for (uint32_t j = 0; j < paddedSizes[1]; ++j) {
        if (i < input->getSizes()[0] && j < input->getSizes()[1])
          inputPaddedReal.getData()[i * paddedSizes[1] + j] =
              input->getData()[i * input->getSizes()[1] + j];
        else
          inputPaddedReal.getData()[i * paddedSizes[1] + j] = constantValue;
      }
    }
  } else if (option == BOUNDARY_OPTION::REPLICATE_PADDING) {
    for (uint32_t i = 0; i < paddedSizes[0]; ++i) {
      uint32_t r = (i < input->getSizes()[0])
                       ? i
                       : ((i < input->getSizes()[0] + centerY)
                              ? (input->getSizes()[0] - 1)
                              : 0);
      for (uint32_t j = 0; j < paddedSizes[1]; ++j) {
        uint32_t c = (j < input->getSizes()[1])
                         ? j
                         : ((j < input->getSizes()[1] + centerX)
                                ? (input->getSizes()[1] - 1)
                                : 0);
        inputPaddedReal.getData()[i * paddedSizes[1] + j] =
            input->getData()[r * input->getSizes()[1] + c];
      }
    }
  }

  // Obtain padded kernel.
  detail::padKernel(&flippedKernel, centerX, centerY, paddedSizes,
                    &kernelPaddedReal);

  detail::_mlir_ciface_corrfft_2d(&inputPaddedReal, &inputPaddedImag,
                                  &kernelPaddedReal, &kernelPaddedImag,
                                  &intermediateReal, &intermediateImag);

  for (uint32_t i = 0; i < output->getSizes()[0]; ++i)
    for (uint32_t j = 0; j < output->getSizes()[1]; ++j)
      output->getData()[i * output->getSizes()[1] + j] =
          inputPaddedReal.getData()[i * paddedSizes[1] + j];
}

// User interface for 2D Rotation.
inline MemRef<float, 2> Rotate2D(Img<float, 2> *input, float angle,
                                 ANGLE_TYPE angleType) {
  float angleRad;

  if (angleType == ANGLE_TYPE::DEGREE)
    angleRad = M_PI * angle / 180;
  else
    angleRad = angle;

  float sinAngle = std::sin(angleRad);
  float cosAngle = std::cos(angleRad);

  int outputRows = std::round(std::abs(input->getSizes()[0] * cosAngle) +
                              std::abs(input->getSizes()[1] * sinAngle));
  int outputCols = std::round(std::abs(input->getSizes()[1] * cosAngle) +
                              std::abs(input->getSizes()[0] * sinAngle));

  intptr_t sizesOutput[2] = {outputRows, outputCols};
  MemRef<float, 2> output(sizesOutput);

  detail::_mlir_ciface_rotate_2d(input, angleRad, &output);

  return output;
}

// User interface for 2D Resize.
inline MemRef<float, 2> Resize2D(Img<float, 2> *input, INTERPOLATION_TYPE type,
                                 std::vector<float> scalingRatios) {
  if (scalingRatios[0] <= 0 || scalingRatios[1] <= 0) {
    throw std::invalid_argument(
        "Please enter positive values of scaling ratios.\n"
        "Note : scaling ratio = "
        "output_image_dimension / input_image_dimension\n");
  }
  std::reverse(scalingRatios.begin(), scalingRatios.end());

  intptr_t outputSize[2] = {static_cast<unsigned>(std::round(
                                input->getSizes()[0] * scalingRatios[0])),
                            static_cast<unsigned>(std::round(
                                input->getSizes()[1] * scalingRatios[1]))};

  scalingRatios[0] = 1 / scalingRatios[0];
  scalingRatios[1] = 1 / scalingRatios[1];

  return detail::Resize2D_Impl(
      input, type, {scalingRatios[1], scalingRatios[0]}, outputSize);
}

// User interface for 2D Resize.
inline MemRef<float, 2> Resize2D(Img<float, 2> *input, INTERPOLATION_TYPE type,
                                 intptr_t outputSize[2]) {
  if (outputSize[0] <= 0 || outputSize[1] <= 0) {
    throw std::invalid_argument(
        "Please enter positive values of output dimensions.\n");
  }
  std::reverse(outputSize, outputSize + 2);

  std::vector<float> scalingRatios(2);
  scalingRatios[1] = input->getSizes()[0] * 1.0f / outputSize[0];
  scalingRatios[0] = input->getSizes()[1] * 1.0f / outputSize[1];

  return detail::Resize2D_Impl(input, type, scalingRatios, outputSize);
}

inline void Erosion2D(Img<float, 2> input, MemRef<float, 2> *kernel,
                      MemRef<float, 2> *output, unsigned int centerX,
                      unsigned int centerY, unsigned int iterations,
                      BOUNDARY_OPTION option, float constantValue = 0) {
  intptr_t outputRows = output->getSizes()[0];
  intptr_t outputCols = output->getSizes()[1];
  intptr_t sizesOutput[2] = {outputRows, outputCols};
  MemRef<float, 2> copymemref(sizesOutput, 256.f);

  if (option == BOUNDARY_OPTION::CONSTANT_PADDING) {
    detail::_mlir_ciface_erosion_2d_constant_padding(
        input, kernel, output, &copymemref, centerX, centerY, iterations,
        constantValue);
  } else if (option == BOUNDARY_OPTION::REPLICATE_PADDING) {
    detail::_mlir_ciface_erosion_2d_replicate_padding(
        input, kernel, output, &copymemref, centerX, centerY, iterations, 0);
  }
}

inline void Dilation2D(Img<float, 2> input, MemRef<float, 2> *kernel,
                       MemRef<float, 2> *output, unsigned int centerX,
                       unsigned int centerY, unsigned int iterations,
                       BOUNDARY_OPTION option, float constantValue = 0) {
  intptr_t outputRows = output->getSizes()[0];
  intptr_t outputCols = output->getSizes()[1];
  intptr_t sizesOutput[2] = {outputRows, outputCols};
  MemRef<float, 2> copymemref(sizesOutput, -1.f);
  if (option == BOUNDARY_OPTION::CONSTANT_PADDING) {
    detail::_mlir_ciface_dilation_2d_constant_padding(
        input, kernel, output, &copymemref, centerX, centerY, iterations,
        constantValue);
  } else if (option == BOUNDARY_OPTION::REPLICATE_PADDING) {
    detail::_mlir_ciface_dilation_2d_replicate_padding(
        input, kernel, output, &copymemref, centerX, centerY, iterations, 0);
  }
}

inline void Opening2D(Img<float, 2> input, MemRef<float, 2> *kernel,
                      MemRef<float, 2> *output, unsigned int centerX,
                      unsigned int centerY, unsigned int iterations,
                      BOUNDARY_OPTION option, float constantValue = 0) {
  intptr_t outputRows = output->getSizes()[0];
  intptr_t outputCols = output->getSizes()[1];

  intptr_t sizesOutput[2] = {outputRows, outputCols};
  MemRef<float, 2> output1(sizesOutput);
  MemRef<float, 2> copymemref(sizesOutput, 256.f);
  MemRef<float, 2> copymemref1(sizesOutput, -1.f);
  if (option == BOUNDARY_OPTION::CONSTANT_PADDING) {
    detail::_mlir_ciface_opening_2d_constant_padding(
        input, kernel, output, &output1, &copymemref, &copymemref1, centerX,
        centerY, iterations, constantValue);
  } else if (option == BOUNDARY_OPTION::REPLICATE_PADDING) {
    detail::_mlir_ciface_opening_2d_replicate_padding(
        input, kernel, output, &output1, &copymemref, &copymemref1, centerX,
        centerY, iterations, 0);
  }
}

inline void Closing2D(Img<float, 2> input, MemRef<float, 2> *kernel,
                      MemRef<float, 2> *output, unsigned int centerX,
                      unsigned int centerY, unsigned int iterations,
                      BOUNDARY_OPTION option, float constantValue = 0) {
  intptr_t outputRows = output->getSizes()[0];
  intptr_t outputCols = output->getSizes()[1];

  intptr_t sizesOutput[2] = {outputRows, outputCols};
  MemRef<float, 2> output1(sizesOutput);
  MemRef<float, 2> copymemref(sizesOutput, -1.f);
  MemRef<float, 2> copymemref1(sizesOutput, 256.f);
  if (option == BOUNDARY_OPTION::CONSTANT_PADDING) {
    detail::_mlir_ciface_closing_2d_constant_padding(
        input, kernel, output, &output1, &copymemref, &copymemref1, centerX,
        centerY, iterations, constantValue);
  } else if (option == BOUNDARY_OPTION::REPLICATE_PADDING) {
    detail::_mlir_ciface_closing_2d_replicate_padding(
        input, kernel, output, &output1, &copymemref, &copymemref1, centerX,
        centerY, iterations, 0);
  }
}

inline void TopHat2D(Img<float, 2> input, MemRef<float, 2> *kernel,
                     MemRef<float, 2> *output, unsigned int centerX,
                     unsigned int centerY, unsigned int iterations,
                     BOUNDARY_OPTION option, float constantValue = 0) {
  intptr_t outputRows = output->getSizes()[0];
  intptr_t outputCols = output->getSizes()[1];
  intptr_t sizesOutput[2] = {outputRows, outputCols};
  MemRef<float, 2> output1(sizesOutput);
  MemRef<float, 2> output2(sizesOutput);
  MemRef<float, 2> input1(sizesOutput);
  MemRef<float, 2> copymemref(sizesOutput, 256.f);
  MemRef<float, 2> copymemref1(sizesOutput, -1.f);
  if (option == BOUNDARY_OPTION::CONSTANT_PADDING) {
    detail::_mlir_ciface_tophat_2d_constant_padding(
        input, kernel, output, &output1, &output2, &input1, &copymemref,
        &copymemref1, centerX, centerY, iterations, constantValue);
  } else if (option == BOUNDARY_OPTION::REPLICATE_PADDING) {
    detail::_mlir_ciface_tophat_2d_replicate_padding(
        input, kernel, output, &output1, &output2, &input1, &copymemref,
        &copymemref1, centerX, centerY, iterations, 0);
  }
}

inline void BottomHat2D(Img<float, 2> input, MemRef<float, 2> *kernel,
                        MemRef<float, 2> *output, unsigned int centerX,
                        unsigned int centerY, unsigned int iterations,
                        BOUNDARY_OPTION option, float constantValue = 0) {
  intptr_t outputRows = output->getSizes()[0];
  intptr_t outputCols = output->getSizes()[1];
  intptr_t sizesOutput[2] = {outputRows, outputCols};
  MemRef<float, 2> output1(sizesOutput);
  MemRef<float, 2> output2(sizesOutput);
  MemRef<float, 2> input1(sizesOutput);
  MemRef<float, 2> copymemref(sizesOutput, -1.f);
  MemRef<float, 2> copymemref1(sizesOutput, 256.f);
  if (option == BOUNDARY_OPTION::CONSTANT_PADDING) {
    detail::_mlir_ciface_bottomhat_2d_constant_padding(
        input, kernel, output, &output1, &output2, &input1, &copymemref,
        &copymemref1, centerX, centerY, iterations, constantValue);
  } else if (option == BOUNDARY_OPTION::REPLICATE_PADDING) {
    detail::_mlir_ciface_bottomhat_2d_replicate_padding(
        input, kernel, output, &output1, &output2, &input1, &copymemref,
        &copymemref1, centerX, centerY, iterations, 0);
  }
}

inline void MorphGrad2D(Img<float, 2> input, MemRef<float, 2> *kernel,
                        MemRef<float, 2> *output, unsigned int centerX,
                        unsigned int centerY, unsigned int iterations,
                        BOUNDARY_OPTION option, float constantValue = 0) {
  intptr_t outputRows = output->getSizes()[0];
  intptr_t outputCols = output->getSizes()[1];
  intptr_t sizesOutput[2] = {outputRows, outputCols};
  MemRef<float, 2> output1(sizesOutput);
  MemRef<float, 2> output2(sizesOutput);
  MemRef<float, 2> input1(sizesOutput);
  MemRef<float, 2> copymemref(sizesOutput, -1.f);
  MemRef<float, 2> copymemref1(sizesOutput, 256.f);
  if (option == BOUNDARY_OPTION::CONSTANT_PADDING) {
    detail::_mlir_ciface_morphgrad_2d_constant_padding(
        input, kernel, output, &output1, &output2, &input1, &copymemref,
        &copymemref1, centerX, centerY, iterations, constantValue);
  } else if (option == BOUNDARY_OPTION::REPLICATE_PADDING) {
    detail::_mlir_ciface_morphgrad_2d_replicate_padding(
        input, kernel, output, &output1, &output2, &input1, &copymemref,
        &copymemref1, centerX, centerY, iterations, 0);
  }
}
} // namespace dip

#endif // FRONTEND_INTERFACES_BUDDY_DIP_DIP
