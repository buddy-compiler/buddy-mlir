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

#include <buddy/Core/Container.h>
#include <buddy/DIP/DIP.h>
#include <buddy/DIP/ImageContainer.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>


using namespace cv;
using namespace std;
using namespace chrono;

#define BUDDY


bool testImplementation(int argc, char *argv[]) {


  // Read as grayscale image.
  Mat image = imread(argv[1], IMREAD_GRAYSCALE);
  if (image.empty()) {
    cout << "Could not read the image: " << argv[1] << endl;
  }

  // Define memref containers.
  Img<float, 2> input(image);
#ifndef BUDDY
  double angle = 30;
  cv::setNumThreads(0);


  // get rotation matrix for rotating the image around its center in pixel coordinates
  Point2f center((image.cols-1)/2.0, (image.rows-1)/2.0);
  Point2f srcp[3] = {Point2f{0, 0}, Point2f{0, 512}, Point2f{512, 0}};
  Point2f dstp[3] = {Point2f{0, 0}, Point2f{0, 256}, Point2f{256, 0}};
  Mat rot = getRotationMatrix2D(center, angle, 1.0);
  // determine bounding rectangle, center not relevant
  Rect2f bbox = RotatedRect(Point2f(), image.size(), angle).boundingRect2f();
  // adjust transformation matrix
  rot.at<double>(0,2) += bbox.width/2.0 - image.cols/2.0;
  rot.at<double>(1,2) += bbox.height/2.0 - image.rows/2.0;
  Mat dst;
#else
  float angleRad = M_PI * 30 / 180;

  float sinAngle = std::sin(angleRad);
  float cosAngle = std::cos(angleRad);

  int outputRows = std::round(std::abs(input.getSizes()[0] * cosAngle) +
                              std::abs(input.getSizes()[1] * sinAngle)) +
                   1;
  int outputCols = std::round(std::abs(input.getSizes()[1] * cosAngle) +
                              std::abs(input.getSizes()[0] * sinAngle)) +
                   1;

  intptr_t sizesOutput[2] = {outputRows, outputCols};
  MemRef<float, 2> output(sizesOutput);
#endif
  // start time count
  auto start = system_clock::now();

#ifdef BUDDY

  dip::detail::_mlir_ciface_rotate_2d(&input, angleRad, &output);
#else
  warpAffine(image, dst, rot, bbox.size(), INTER_NEAREST);
#endif
  // end time count
  auto end = system_clock::now();

  // output time
  auto duration = duration_cast<microseconds>(end - start);
  cout << (double)duration.count() * microseconds::period::num / microseconds::period::den << endl;

  // Define a cv::Mat with the output of Rotate2D.
#ifdef BUDDY
  Mat outputImageRotate2D(output.getSizes()[0], output.getSizes()[1], CV_32FC1,
                          output.getData());
  imwrite(argv[2], outputImageRotate2D);
#else
  imwrite(argv[2], dst);
#endif

  return 1;
}

int main(int argc, char *argv[]) {
  testImplementation(argc, argv);

  return 0;
}
