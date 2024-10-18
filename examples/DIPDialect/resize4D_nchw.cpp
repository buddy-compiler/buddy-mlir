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
#include <buddy/DIP/ImgContainer.h>
#include <iostream>
#include <math.h>

using namespace std;

void testImplementation(int argc, char *argv[]) {
  // Read as colar image.
  dip::Image<float, 4> inputBatch(argv[1], dip::DIP_RGB);

  // Note : Both values in output image dimensions and scaling ratios must be
  // positive numbers.
  MemRef<float, 4> output = dip::Resize4D_NCHW(
      &inputBatch, dip::INTERPOLATION_TYPE::BILINEAR_INTERPOLATION,
      {1, 3, 224, 224} /*{image_cols, image_rows}*/);

  // Define Img with the output of Resize4D.
  intptr_t outSizes[4] = {output.getSizes()[0], output.getSizes()[1],
                          output.getSizes()[2], output.getSizes()[3]};

  dip::Image<float, 4> outputImageResize4D(output.getData(), outSizes);

  dip::imageWrite(argv[2], outputImageResize4D);

  return;
}

int main(int argc, char *argv[]) {
  testImplementation(argc, argv);
  return 0;
}
