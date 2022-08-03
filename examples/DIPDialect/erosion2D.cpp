//===- erosion2D.cpp - Example of buddy-opt tool ----------------------===//
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

      //  std::cout << img1 << "\n\n";
       // std::cout << img2 << "\n\n";
        return 0;
      }
    }
  }
  return 1;
}

bool testImplementation(int argc, char *argv[], std::ptrdiff_t x,
                        std::ptrdiff_t y, std::ptrdiff_t boundaryOption) {
  // Read as grayscale image.
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





  Mat kernel1 = Mat(3, 3, CV_32FC1, laplacianKernelAlign);


  // Call the MLIR Erosion2D function.
  dip::Erosion2D(&input, &kernel, &output1, x, y,
              dip::BOUNDARY_OPTION::REPLICATE_PADDING,dip::STRUCTURING_TYPE::FLAT,0);

  // Define a cv::Mat with the output of Erosion2D.
  Mat outputImageReplicatePadding_flat(sizesOutput[0], sizesOutput[1], CV_32FC1,
                                  output1.getData());
  imwrite(argv[2], outputImageReplicatePadding_flat);

  Mat o1 = imread(argv[2], IMREAD_GRAYSCALE);
  Mat opencvConstantPaddingflat, opencvReplicatePaddingflat, opencvConstantPaddingnonflat, opencvReplicatePaddingnonflat;
  erode(image, opencvReplicatePaddingflat, kernel1, cv::Point(x,y), 1, cv::BORDER_CONSTANT, 0);
  imwrite(argv[3], opencvReplicatePaddingflat);

  if (!testImages(o1, opencvReplicatePaddingflat)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }


  // Call the MLIR Erosion2D function.
 dip::Erosion2D(&input, &kernel, &output2, x, y,
              dip::BOUNDARY_OPTION::CONSTANT_PADDING,dip::STRUCTURING_TYPE::FLAT,0.0);

  // Define a cv::Mat with the output of Erosion2D.
  Mat outputImageConstantPadding_flat(sizesOutput[0], sizesOutput[1], CV_32FC1,
                                 output2.getData());
  imwrite(argv[4], outputImageConstantPadding_flat);

  Mat o2 = imread(argv[4], IMREAD_GRAYSCALE);
 erode(image, opencvConstantPaddingflat, kernel1, cv::Point(x, y), 1, cv::BORDER_CONSTANT, 0.0);
 imwrite(argv[5], opencvReplicatePaddingflat);

  if (!testImages(o2, opencvConstantPaddingflat)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }


dip::Erosion2D(&input, &kernel, &output3, x, y,
            dip::BOUNDARY_OPTION::CONSTANT_PADDING, dip::STRUCTURING_TYPE::NONFLAT,1.0);
    Mat outputImageConstantPadding_nonflat(sizesOutput[0], sizesOutput[1], CV_32FC1,
                                 output3.getData());
                                 imwrite(argv[6], outputImageConstantPadding_nonflat);
                                 Mat o3 = imread(argv[6], IMREAD_GRAYSCALE);
          erode(image, opencvConstantPaddingnonflat, kernel1, cv::Point(x, y), 1, cv::BORDER_CONSTANT, 0.0);
           imwrite(argv[7], opencvReplicatePaddingflat);
             if (!testImages(o3, opencvConstantPaddingnonflat)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }


dip::Erosion2D(&input, &kernel, &output4, x, y,
            dip::BOUNDARY_OPTION::REPLICATE_PADDING, dip::STRUCTURING_TYPE::NONFLAT,1.0);
    Mat outputImageReplicatePadding_nonflat(sizesOutput[0], sizesOutput[1], CV_32FC1,
                                 output4.getData());
                                 imwrite(argv[8], outputImageReplicatePadding_nonflat);
                                 Mat o4 = imread(argv[8], IMREAD_GRAYSCALE);
          
erode(image, opencvReplicatePaddingnonflat, kernel1, cv::Point(x, y), 1, cv::BORDER_REPLICATE, 0.0);
imwrite(argv[9], opencvReplicatePaddingnonflat);
             if (!testImages(o4, opencvReplicatePaddingnonflat)) {
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