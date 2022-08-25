//===- tophat2D.cpp - Example of buddy-opt tool ----------------------===//
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
// This file implements  2D opening examples with dip.opening_2d operation.
// The dip.dilation_2d operation will be compiled into an object file with the
// buddy-opt tool.
// This file will be linked with the object file to generate the executable
// file.
//
//===----------------------------------------------------------------------===//

#include <opencv2/opencv.hpp>

#include "../ConvOpt/kernels.h"
#include "Interface/buddy/core/ImageContainer.h"
#include <Interface/buddy/dip/dip.h>
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;

bool testImages(cv::Mat img1, cv::Mat img2) { 
  if (img1.rows != img2.rows || img1.cols != img2.cols) {
    std::cout << "Dimensions not equal\n";
    return 0;
  }

  for (std::ptrdiff_t i = 0; i < img1.cols; ++i) {
    for (std::ptrdiff_t j = 0; j < img1.rows; ++j) {
      if (img1.at<uchar>(i, j) != img2.at<uchar>(i, j)) {
        std::cout << "Pixels not equal at : (" << i << "," << j << ")\n";
        std::cout << (int)img1.at<uchar>(i, j) << "\n";
        std::cout << (int)img2.at<uchar>(i, j) << "\n\n";

       std::cout << img1 << "\n\n";
       std::cout << img2 << "\n\n";
        return 0;
      }
    }
  }
  return 1;
}

bool testImplementation(int argc, char *argv[], std::ptrdiff_t x,
                        std::ptrdiff_t y, std::ptrdiff_t boundaryOption) 
                        {
                           Mat image = imread(argv[1], IMREAD_GRAYSCALE);
  if (image.empty()) {
    cout << "Could not read the image: " << argv[1] << endl;
  }


  // Define the kernel.
  float *kernelAlign = laplacianKernelAlign;
  int kernelRows = laplacianKernelRows;
  int kernelCols = laplacianKernelCols;


  // Define sizes and strides.
  intptr_t sizesKernel[2] = {kernelRows, kernelCols};
  intptr_t sizesOutput[2] = {image.rows, image.cols};

    // Define memref containers.
  Img<float, 2> input(image);
  MemRef<float, 2> kernel(kernelAlign, sizesKernel);
  MemRef<float, 2> output1(sizesOutput);
  MemRef<float, 2> output2(sizesOutput);
  MemRef<float, 2> output3(sizesOutput);
  MemRef<float, 2> output4(sizesOutput);


 cv::Mat kernel1 = Mat(3, 3, CV_32FC1, laplacianKernelAlign);

  dip::BottomHat2D(&input, &kernel, &output1, x, y,
              dip::BOUNDARY_OPTION::CONSTANT_PADDING,0);
   Mat outputImageReplicatePadding_flat(sizesOutput[0], sizesOutput[1], CV_32FC1,
                                  output1.getData());
  imwrite(argv[2], outputImageReplicatePadding_flat);

  Mat o1 = imread(argv[2], IMREAD_GRAYSCALE);
  // cv::Mat to store output of the permutations of dilation op
  Mat opencvConstantPaddingflat, opencvReplicatePaddingflat, opencvConstantPaddingnonflat, opencvReplicatePaddingnonflat;
  morphologyEx(image, opencvReplicatePaddingflat,6, kernel1, cv::Point(x,y), 1, 0, 0);
  imwrite(argv[3], opencvReplicatePaddingflat);

  if (!testImages(o1, opencvReplicatePaddingflat)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }
dip::TopHat2D(&input, &kernel, &output2, x, y,
              dip::BOUNDARY_OPTION::CONSTANT_PADDING,0.0);

  // Define a cv::Mat with the output of Dilation2D.
  Mat outputImageConstantPaddingflat(sizesOutput[0], sizesOutput[1], CV_32FC1,
                                 output2.getData());
  imwrite(argv[4], outputImageConstantPaddingflat);

  Mat o2 = imread(argv[4], IMREAD_GRAYSCALE);
 morphologyEx(image, opencvConstantPaddingflat, 3, kernel1, cv::Point(x, y), 1, 0, 0.0);
 imwrite(argv[5], opencvConstantPaddingflat);

  if (!testImages(o2, opencvConstantPaddingflat)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }

return 1;
                       }

int main(int argc, char *argv[]) {
  std::ptrdiff_t x = 1;
  std::ptrdiff_t y =1;
testImplementation(argc, argv, x, y, 0);
  return 0;
}