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
#include "buddy/DIP/ImgContainer.h"
#include <iostream>
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

// Available formats for 4D images.
enum class IMAGE_FORMAT { NHWC, NCHW };

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

// Declare the Rotate4D C interface.
void _mlir_ciface_rotate_4d_nhwc(Img<float, 4> *input, float angleValue,
                                 MemRef<float, 4> *output);

void _mlir_ciface_rotate_4d_nchw(Img<float, 4> *input, float angleValue,
                                 MemRef<float, 4> *output);

// Declare the Resize2D C interface.
void _mlir_ciface_resize_2d_nearest_neighbour_interpolation(
    Img<float, 2> *input, float horizontalScalingFactor,
    float verticalScalingFactor, MemRef<float, 2> *output);

// Declare the Resize4D C interface.
void _mlir_ciface_resize_4d_nhwc_nearest_neighbour_interpolation(
    Img<float, 4> *input, float horizontalScalingFactor,
    float verticalScalingFactor, MemRef<float, 4> *output);

void _mlir_ciface_resize_4d_nchw_nearest_neighbour_interpolation(
    dip::Image<float, 4> *input, float horizontalScalingFactor,
    float verticalScalingFactor, MemRef<float, 4> *output);

void _mlir_ciface_resize_2d_bilinear_interpolation(
    Img<float, 2> *input, float horizontalScalingFactor,
    float verticalScalingFactor, MemRef<float, 2> *output);

// Declare the Resize4D C interface.
void _mlir_ciface_resize_4d_nhwc_bilinear_interpolation(
    Img<float, 4> *input, float horizontalScalingFactor,
    float verticalScalingFactor, MemRef<float, 4> *output);

void _mlir_ciface_resize_4d_nchw_bilinear_interpolation(
    dip::Image<float, 4> *input, float horizontalScalingFactor,
    float verticalScalingFactor, MemRef<float, 4> *output);

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

template <typename T, int D>
void Transpose(MemRef<T, D> *output, MemRef<T, D> *input,
               const std::vector<int> &axes) {
  std::vector<intptr_t> inputDims(D);
  for (int i = 0; i < D; ++i) {
    inputDims[i] = input->getSizes()[i];
  }

  std::vector<intptr_t> outputDims(D);
  for (int i = 0; i < D; ++i) {
    outputDims[i] = inputDims[axes[i]];
  }

  const T *inputData = input->getData();
  T *outputData = output->getData();

  std::vector<intptr_t> inputStrides(D);
  inputStrides[D - 1] = 1;
  for (int i = D - 2; i >= 0; --i) {
    inputStrides[i] = inputStrides[i + 1] * inputDims[i + 1];
  }

  std::vector<intptr_t> outputStrides(D);
  outputStrides[D - 1] = 1;
  for (int i = D - 2; i >= 0; --i) {
    outputStrides[i] = outputStrides[i + 1] * outputDims[i + 1];
  }

  std::vector<intptr_t> indices(D, 0);
  std::vector<intptr_t> outputIndices(D, 0);

  while (true) {
    intptr_t inputIndex = 0;
    intptr_t outputIndex = 0;
    for (int i = 0; i < D; ++i) {
      inputIndex += indices[i] * inputStrides[i];
      outputIndex += indices[axes[i]] * outputStrides[i];
    }
    outputData[outputIndex] = inputData[inputIndex];

    int i = D - 1;
    while (i >= 0) {
      indices[i]++;
      if (indices[i] < inputDims[i])
        break;
      indices[i] = 0;
      i--;
    }
    if (i < 0)
      break;
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

// Helper function for applying 4D resize operation on images.
inline MemRef<float, 4> Resize4D_NHWC_Impl(Img<float, 4> *input,
                                           INTERPOLATION_TYPE type,
                                           std::vector<float> scalingRatios,
                                           intptr_t outputSize[4]) {
  MemRef<float, 4> output(outputSize);

  if (type == INTERPOLATION_TYPE::NEAREST_NEIGHBOUR_INTERPOLATION) {
    detail::_mlir_ciface_resize_4d_nhwc_nearest_neighbour_interpolation(
        input, scalingRatios[0], scalingRatios[1], &output);
  } else if (type == INTERPOLATION_TYPE::BILINEAR_INTERPOLATION) {
    detail::_mlir_ciface_resize_4d_nhwc_bilinear_interpolation(
        input, scalingRatios[0], scalingRatios[1], &output);
  } else {
    throw std::invalid_argument(
        "Please chose a supported type of interpolation "
        "(Nearest neighbour interpolation or Bilinear interpolation)\n");
  }

  return output;
}

inline MemRef<float, 4> Resize4D_NCHW_Impl(dip::Image<float, 4> *input,
                                           INTERPOLATION_TYPE type,
                                           std::vector<float> scalingRatios,
                                           intptr_t outputSize[4]) {
  MemRef<float, 4> output(outputSize);

  if (type == INTERPOLATION_TYPE::NEAREST_NEIGHBOUR_INTERPOLATION) {
    detail::_mlir_ciface_resize_4d_nchw_nearest_neighbour_interpolation(
        input, scalingRatios[0], scalingRatios[1], &output);
  } else if (type == INTERPOLATION_TYPE::BILINEAR_INTERPOLATION) {
    detail::_mlir_ciface_resize_4d_nchw_bilinear_interpolation(
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

inline MemRef<float, 4> Rotate4D(Img<float, 4> *input, float angle,
                                 ANGLE_TYPE angleType, IMAGE_FORMAT format) {
  float angleRad;

  if (angleType == ANGLE_TYPE::DEGREE)
    angleRad = M_PI * angle / 180;
  else
    angleRad = angle;

  float sinAngle = std::sin(angleRad);
  float cosAngle = std::cos(angleRad);

  int outputRows, outputCols;
  intptr_t sizesOutput[4];
  if (format == IMAGE_FORMAT::NHWC) {
    outputRows = std::round(std::abs(input->getSizes()[1] * cosAngle) +
                            std::abs(input->getSizes()[2] * sinAngle));
    outputCols = std::round(std::abs(input->getSizes()[2] * cosAngle) +
                            std::abs(input->getSizes()[1] * sinAngle));
    sizesOutput[0] = input->getSizes()[0];
    sizesOutput[1] = outputRows;
    sizesOutput[2] = outputCols;
    sizesOutput[3] = input->getSizes()[3];
  } else {
    //  format == IMAGE_FORMAT::NCHW
    outputRows = std::round(std::abs(input->getSizes()[2] * cosAngle) +
                            std::abs(input->getSizes()[3] * sinAngle));
    outputCols = std::round(std::abs(input->getSizes()[3] * cosAngle) +
                            std::abs(input->getSizes()[2] * sinAngle));
    sizesOutput[0] = input->getSizes()[0];
    sizesOutput[1] = input->getSizes()[1];
    sizesOutput[2] = outputRows;
    sizesOutput[3] = outputCols;
  }

  MemRef<float, 4> output(sizesOutput);
  if (format == IMAGE_FORMAT::NHWC) {
    detail::_mlir_ciface_rotate_4d_nhwc(input, angleRad, &output);
  } else {
    // format == FORMAT_4D_IMAGE::NCHW
    detail::_mlir_ciface_rotate_4d_nchw(input, angleRad, &output);
  }

  return output;
}

// User interface for 2D Resize.
inline MemRef<float, 2> Resize2D(Img<float, 2> *input, INTERPOLATION_TYPE type,
                                 std::vector<uint> size) {
  if (size.size() != 2) {
    throw std::invalid_argument("Dimension of an image should be 2\n");
  }
  intptr_t outputSize[2] = {size[0], size[1]};
  return detail::Resize2D_Impl(input, type,
                               {(float)input->getSizes()[0] / (float)size[0],
                                (float)input->getSizes()[1] / (float)size[1]},
                               outputSize);
}

// User interface for 4D Resize.
inline MemRef<float, 4> Resize4D_NHWC(Img<float, 4> *input,
                                      INTERPOLATION_TYPE type,
                                      std::vector<uint> size) {
  if (size.size() != 4) {
    throw std::invalid_argument("Dimension of an image should be 4\n");
  }
  intptr_t outputSize[4] = {size[0], size[1], size[2], size[3]};
  return detail::Resize4D_NHWC_Impl(
      input, type,
      {(float)input->getSizes()[1] / (float)size[1],
       (float)input->getSizes()[2] / (float)size[2]},
      outputSize);
}

inline MemRef<float, 4> Resize4D_NCHW(dip::Image<float, 4> *input,
                                      INTERPOLATION_TYPE type,
                                      std::vector<uint> size) {
  if (size.size() != 4) {
    throw std::invalid_argument("Dimension of an image should be 4\n");
  }
  intptr_t outputSize[4] = {size[0], size[1], size[2], size[3]};
  return detail::Resize4D_NCHW_Impl(
      input, type,
      {(float)input->getSizes()[2] / (float)size[2],
       (float)input->getSizes()[3] / (float)size[3]},
      outputSize);
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
