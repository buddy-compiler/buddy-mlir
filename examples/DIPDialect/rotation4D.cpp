//====- rotation4D.cpp - Example of buddy-opt tool ===========================//
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
// This file implements a 4D rotation example with dip.rotate_4d operation.
// The dip.rotate_4d operation will be compiled into an object file with the
// buddy-opt tool.
// This file will be linked with the object file to generate the executable
// file.
//
//===----------------------------------------------------------------------===//

#include "buddy/DIP/imgcodecs/loadsave.h"
#include <buddy/Core/Container.h>
#include <buddy/DIP/DIP.h>
#include <buddy/DIP/ImageContainer.h>
#include <iostream>
#include <math.h>

using namespace std;

bool testImplementation(int argc, char *argv[]) {
  const int inputBatch = 1;

  // Read as color image in [HWC] format
  Img<float, 3> input = dip::imread<float, 3>(argv[1], dip::IMGRD_COLOR);
  const int inputHeight = input.getSizes()[0];
  const int inputWidth = input.getSizes()[1];
  const int inputChannels = input.getSizes()[2];
  const int inputStride = inputHeight * inputWidth * inputChannels;

  //  Image format is [NHWC]
  intptr_t inputSizes_NHWC[4] = {inputBatch, inputHeight, inputWidth,
                                 inputChannels};
  Img<float, 4> inputImages_NHWC(inputSizes_NHWC);

  auto imagePtr = inputImages_NHWC.getData();
  memcpy(imagePtr, input.getData(), inputStride * sizeof(float));
  for (int i = 1; i < inputBatch; i++) {
    Img<float, 3> input = dip::imread<float, 3>(argv[1], dip::IMGRD_COLOR);
    memcpy(imagePtr + i * inputStride, input.getData(),
           inputStride * sizeof(float));
  }

  MemRef<float, 4> output = dip::Rotate4D(
      &inputImages_NHWC, 30, dip::ANGLE_TYPE::DEGREE, dip::IMAGE_FORMAT::NHWC);

  const int outoutHeight = output.getSizes()[1];
  const int outputWidth = output.getSizes()[2];
  intptr_t outputSizes_NHWC[4] = {inputBatch, outoutHeight, outputWidth,
                                  inputChannels};
  const int outputStride = outoutHeight * outputWidth * inputChannels;
  Img<float, 4> outputImages_NHWC(output.getData(), outputSizes_NHWC);

  for (int i = 0; i < inputBatch; i++) {
    intptr_t imageSizes[3] = {outoutHeight, outputWidth, inputChannels};
    Img<float, 3> outputImage(outputImages_NHWC.getData() + i * outputStride,
                              imageSizes);
    dip::imwrite(argv[2], outputImage);
  }

  // Image Format is [NCHW]
  // Rearrange memory layout
  intptr_t inputSizes_NCHW[4] = {inputBatch, inputChannels, inputHeight,
                                 inputWidth};
  Img<float, 4> inputImages_NCHW(inputSizes_NCHW);
  dip::detail::Transpose<float, 4>(&inputImages_NCHW, &inputImages_NHWC,
                                   {0, 3, 1, 2});
  output = dip::Rotate4D(&inputImages_NCHW, 30, dip::ANGLE_TYPE::DEGREE,
                         dip::IMAGE_FORMAT::NCHW);

  intptr_t outputSizes_NCHW[4] = {inputBatch, inputChannels, outoutHeight,
                                  outputWidth};
  Img<float, 4> outputImages_NCHW(output.getData(), outputSizes_NCHW);
  dip::detail::Transpose<float, 4>(&outputImages_NHWC, &outputImages_NCHW,
                                   {0, 2, 3, 1});

  for (int i = 0; i < inputBatch; i++) {
    intptr_t imageSizes[3] = {outoutHeight, outputWidth, inputChannels};
    Img<float, 3> outputImage(outputImages_NHWC.getData() + i * outputStride,
                              imageSizes);
    dip::imwrite(argv[2], outputImage);
  }

  return 1;
}

int main(int argc, char *argv[]) {
  testImplementation(argc, argv);

  return 0;
}
