//====- rotation3D.cpp - Example of buddy-opt tool ===========================//
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
// This file demonstrates an example of performing a 3D rotation using the
// perspective_transform_3d operator. The perspective_transform_3d operator will
// be compiled into an object file using the buddy-opt tool.
// This file will be linked with the object file to generate the executable
// file.
//
//===----------------------------------------------------------------------===//
#include <buddy/DIP/DIP.h>
#include <buddy/DIP/ImageContainer.h>
#include <buddy/DIP/imgcodecs/loadsave.h>
#include <opencv2/core.hpp>

bool testImplementation(int argc, char *argv[]) {
  // Read as grayscale image.
  Img<float, 2> input = dip::imread<float, 2>(argv[1], dip::IMGRD_GRAYSCALE);

  MemRef<float, 2> output =
      dip::Rotate3D(&input, 0, 0, 0, 90, dip::ANGLE_TYPE::DEGREE);

  intptr_t sizes[2] = {output.getSizes()[0], output.getSizes()[1]};
  Img<float, 2> tmp(output.getData(), sizes);

  dip::imwrite(argv[2], tmp);

  return 1;
}

void showImageInPopupWindow(int argc, char *argv[]) {
  // Read as grayscale image.
  Img<float, 2> input = dip::imread<float, 2>(argv[1], dip::IMGRD_GRAYSCALE);

  for (int c = 0, i = 0; c != 27; i++) {
    MemRef<float, 2> output =
        dip::Rotate3D(&input, i, 0, 0, 90, dip::ANGLE_TYPE::DEGREE);

    intptr_t height = output.getSizes()[0], width = output.getSizes()[1];

    cv::Mat img(height, width, CV_32FC1, output.getData());
    img.convertTo(img, CV_8UC1);
    cv::imshow("Window", img);
    c = cv::waitKey(10);
  }
}

int main(int argc, char *argv[]) {
  // testImplementation(argc, argv);
  showImageInPopupWindow(argc, argv);

  return 0;
}
