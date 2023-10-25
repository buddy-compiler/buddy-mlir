//====- resize2D.cpp - Example of buddy-opt tool =============================//
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
// This file implements a 2D resize example with dip.resize_2d operation.
// The dip.resize_2d operation will be compiled into an object file with the
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
  // Read as grayscale image and Define memref container for image.
  Img<float, 2> input = dip::imread<float, 2>(argv[1], dip::IMGRD_GRAYSCALE);

  intptr_t outputSize[2] = {250, 100}; // {image_cols, image_rows}
  std::vector<float> scalingRatios = {
      0.25, 0.1}; // {col_scaling_ratio, row_scaling_ratio}

  // dip::Resize2D() can be called with either scaling ratios
  // (Output image dimension / Input image dimension) for both dimensions or
  // the output image dimensions.
  // Note : Both values in output image dimensions and scaling ratios must be
  // positive numbers.

  MemRef<float, 2> output = dip::Resize2D(
      &input, dip::INTERPOLATION_TYPE::NEAREST_NEIGHBOUR_INTERPOLATION,
      outputSize);
  // MemRef<float, 2> output = dip::Resize2D(
  //     &input, dip::INTERPOLATION_TYPE::BILINEAR_INTERPOLATION, outputSize);

  // MemRef<float, 2> output = dip::Resize2D(
  //     &input, dip::INTERPOLATION_TYPE::NEAREST_NEIGHBOUR_INTERPOLATION,
  //     scalingRatios);
  // MemRef<float, 2> output = dip::Resize2D(
  //     &input, dip::INTERPOLATION_TYPE::BILINEAR_INTERPOLATION,
  //     scalingRatios);

  // Define Img with the output of Resize2D.
  intptr_t sizes[2] = {output.getSizes()[0], output.getSizes()[1]};

  Img<float, 2> outputImageResize2D(output.getData(),sizes);

  dip::imwrite(argv[2], outputImageResize2D);

  return 1;
}

int main(int argc, char *argv[]) {
  testImplementation(argc, argv);

  return 0;
}
