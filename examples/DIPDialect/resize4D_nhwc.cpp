//====- resize4D.cpp - Example of buddy-opt tool =============================//
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
// This file implements a 4D resize example with dip.resize_4d operation.
// The dip.resize_4d operation will be compiled into an object file with the
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

void testImplementation(int argc, char *argv[]) {
  // Read as colar image.
  Img<float, 3> input = dip::imread<float, 3>(argv[1], dip::IMGRD_COLOR);

  intptr_t sizes[4] = {1, input.getSizes()[0], input.getSizes()[1],
                       input.getSizes()[2]};
  Img<float, 4> inputBatch(input.getData(), sizes);

  // Note : Both values in output image dimensions and scaling ratios must be
  // positive numbers.
  MemRef<float, 4> output = dip::Resize4D_NHWC(
      &inputBatch, dip::INTERPOLATION_TYPE::BILINEAR_INTERPOLATION,
      {1, 224, 224, 3} /*{image_cols, image_rows}*/);

  // Define Img with the output of Resize4D.
  intptr_t outSizes[3] = {output.getSizes()[1], output.getSizes()[2],
                          output.getSizes()[3]};

  Img<float, 3> outputImageResize4D(output.getData(), outSizes);

  dip::imwrite(argv[2], outputImageResize4D);

  return;
}

int main(int argc, char *argv[]) {
  testImplementation(argc, argv);
  return 0;
}
