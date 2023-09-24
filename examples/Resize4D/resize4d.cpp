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

#include <buddy/Core/Container.h>
#include <buddy/DIP/DIP.h>
#include <buddy/DIP/ImageContainer.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void testImplementation(int argc, char *argv[]) {
  // Read as grayscale image.
  Mat image = imread(argv[1], IMREAD_COLOR);
  if (image.empty()) {
    cout << "Could not read the image: " << argv[1] << endl;
  }

  // Define memref container for image.
  Img<float, 4> input(image);
  
  // Note : Both values in output image dimensions and scaling ratios must be
  // positive numbers.

  MemRef<float, 4> output = dip::Resize4D(
      &input, dip::INTERPOLATION_TYPE::NEAREST_NEIGHBOUR_INTERPOLATION,
      {1 , 224 , 224 , 3} /*{image_cols, image_rows}*/);
  
  Mat outputImageResize4D(output.getSizes()[1], output.getSizes()[2], CV_32FC1,
                          output.getData());

  imwrite(argv[2], outputImageResize4D);

  return;
}

int main(int argc, char *argv[]) {
  testImplementation(argc, argv);
  return 0;
}
