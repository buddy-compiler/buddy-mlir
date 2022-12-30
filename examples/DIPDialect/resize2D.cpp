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

#include <opencv2/opencv.hpp>
#include "Interface/buddy/core/Container.h"
#include "Interface/buddy/core/ImageContainer.h"
#include <Interface/buddy/dip/dip.h>
#include <iostream>

using namespace cv;
using namespace std;

bool testImplementation(int argc, char *argv[]) {
  // Read as grayscale image.
  Mat image = imread(argv[1], IMREAD_GRAYSCALE);
  if (image.empty()) {
    cout << "Could not read the image: " << argv[1] << endl;
  }

  // Define memref container for image.
  Img<float, 2> input(image);

  size_t outputSize[2] = {250, 250};
  std::vector<float> scalingRatios = {4, 4};

  // dip::Resize2D() can be called with either scaling ratios
  // (Input image dimension / Output image dimension) for both dimensions or
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

  // Define cv::Mat with the output of Resize2D.
  Mat outputImageResize2D(output.getSizes()[0], output.getSizes()[1], CV_32FC1,
                          output.getData());

  imwrite(argv[2], outputImageResize2D);

  return 1;
}

int main(int argc, char *argv[]) {
  testImplementation(argc, argv);

  return 0;
}
