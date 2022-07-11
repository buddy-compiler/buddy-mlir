//====- rotation2D.cpp - Example of buddy-opt tool ===========================//
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
// This file implements a 2D rotation example with dip.rotate_2d operation.
// The dip.rotate_2d operation will be compiled into an object file with the
// buddy-opt tool.
// This file will be linked with the object file to generate the executable
// file.
//
//===----------------------------------------------------------------------===//

#include <opencv2/opencv.hpp>

#include <Interface/buddy/dip/dip.h>
#include <Interface/buddy/dip/memref.h>
#include <iostream>

using namespace cv;
using namespace std;

bool testImplementation(int argc, char *argv[]) {
  // Read as grayscale image.
  Mat image = imread(argv[1], IMREAD_GRAYSCALE);
  if (image.empty()) {
    cout << "Could not read the image: " << argv[1] << endl;
  }

  // Define the input with the image.
  int inputSize = image.rows * image.cols;
  float *inputAlign = (float *)malloc(inputSize * sizeof(float));
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      inputAlign[image.rows * i + j] = (float)image.at<uchar>(i, j);
    }
  }

  // Define allocated, sizes, and strides fields for the MemRef_descriptor.
  float *allocated = (float *)malloc(1 * sizeof(float));
  intptr_t sizesInput[2] = {image.rows, image.cols};
  intptr_t stridesInput[2] = {image.rows, image.cols};

  // Define memref descriptors.
  MemRef_descriptor input =
      MemRef_Descriptor(allocated, inputAlign, 0, sizesInput, stridesInput);
  MemRef_descriptor output = dip::Rotate2D(input, 45, dip::ANGLE_TYPE::DEGREE);

  // Define a cv::Mat with the output of Rotate2D.
  Mat outputImageRotate2D(output->sizes[0], output->sizes[1], CV_32FC1,
                          output->aligned);
  imwrite(argv[2], outputImageRotate2D);

  free(input);
  free(inputAlign);

  return 1;
}

int main(int argc, char *argv[]) {
  testImplementation(argc, argv);

  return 0;
}
