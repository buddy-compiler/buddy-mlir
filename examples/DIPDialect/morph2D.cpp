//===- morph2D.cpp - Example of buddy-opt tool ----------------------===//
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
// This file implements examples of morphological transformations of the DIP
// Dialect. This file will be linked with the object file to generate the
// executable file.
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
                        std::ptrdiff_t y, std::ptrdiff_t boundaryOption) {
  // Read as grayscale image.
  Mat image = imread(argv[1], IMREAD_GRAYSCALE);
  if (image.empty()) {
    cout << "Could not read the image: " << argv[1] << endl;
  }

  // Define the kernel.
  float *kernelAlign = ellipseKernelAlign7x7;
  int kernelRows = ellipseKernelRows7x7;
  int kernelCols = ellipseKernelCols7x7;

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
  MemRef<float, 2> output5(sizesOutput);
  MemRef<float, 2> output6(sizesOutput);
  MemRef<float, 2> output7(sizesOutput);
  MemRef<float, 2> output8(sizesOutput);
  MemRef<float, 2> output9(sizesOutput);
  MemRef<float, 2> output10(sizesOutput);
  MemRef<float, 2> output11(sizesOutput);
  MemRef<float, 2> output12(sizesOutput);
  MemRef<float, 2> output13(sizesOutput);
  MemRef<float, 2> output14(sizesOutput);

  // kernel for morphological transformations.
  Mat kernel1 = cv::getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
  // Call the MLIR Dilation2D function.
  dip::Dilation2D(&input, &kernel, &output1, x, y, 3,
                  dip::BOUNDARY_OPTION::REPLICATE_PADDING, 0);

  // Define a cv::Mat with the output of Dilation2D.
  Mat outputImageReplicatePaddingdilation(sizesOutput[0], sizesOutput[1],
                                          CV_32FC1, output1.getData());
  imwrite(argv[2], outputImageReplicatePaddingdilation);

  Mat o1 = imread(argv[2], IMREAD_GRAYSCALE);
  // cv::Mat to store output of the permutations of dilation op
  Mat opencvReplicatePaddingdilation;
  dilate(image, opencvReplicatePaddingdilation, kernel1, cv::Point(1, 1), 3,
         cv::BORDER_REPLICATE, 0);

  if (!testImages(o1, opencvReplicatePaddingdilation)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }

  // Call the MLIR Dilation2D function.
  dip::Dilation2D(&input, &kernel, &output2, x, y, 5,
                  dip::BOUNDARY_OPTION::CONSTANT_PADDING, 0.0);
  // Define a cv::Mat with the output of Dilation2D.
  Mat outputImageConstantPaddingdilation(sizesOutput[0], sizesOutput[1],
                                         CV_32FC1, output2.getData());
  imwrite(argv[3], outputImageConstantPaddingdilation);
  Mat o2 = imread(argv[3], IMREAD_GRAYSCALE);
  // Define a cv::Mat for storing output of Opencv's dilate method.
  Mat opencvConstantPaddingdilation;
  cv::dilate(image, opencvConstantPaddingdilation, kernel1, cv::Point(x, y), 5,
             cv::BORDER_CONSTANT, 0.0);
  if (!testImages(o2, opencvConstantPaddingdilation)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }

  // Call the MLIR Erosion2D function.
  dip::Erosion2D(&input, &kernel, &output3, x, y, 1,
                 dip::BOUNDARY_OPTION::REPLICATE_PADDING, 0);

  // Define a cv::Mat with the output of Erosion2D.
  Mat outputImageReplicatePaddingerosion(sizesOutput[0], sizesOutput[1],
                                         CV_32FC1, output3.getData());
  imwrite(argv[4], outputImageReplicatePaddingerosion);

  Mat o3 = imread(argv[4], IMREAD_GRAYSCALE);
  // Define a cv::mat for storing output of cv::erode
  Mat opencvReplicatePaddingerosion;
  erode(image, opencvReplicatePaddingerosion, kernel1, cv::Point(x, y), 1,
        cv::BORDER_REPLICATE, 0.0);

  if (!testImages(o3, opencvReplicatePaddingerosion)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }

  // Call the MLIR Erosion2D function.
  dip::Erosion2D(&input, &kernel, &output4, x, y, 2,
                 dip::BOUNDARY_OPTION::CONSTANT_PADDING, 0.0);

  // Define a cv::Mat with the output of Erosion2D.
  Mat outputImageConstantPaddingerosion(sizesOutput[0], sizesOutput[1],
                                        CV_32FC1, output4.getData());
  imwrite(argv[5], outputImageConstantPaddingerosion);

  Mat o4 = imread(argv[5], IMREAD_GRAYSCALE);
  Mat opencvConstantPaddingerosion;
  erode(image, opencvConstantPaddingerosion, kernel1, cv::Point(x, y), 2,
        cv::BORDER_CONSTANT, 0.0);

  if (!testImages(o4, opencvConstantPaddingerosion)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }

  dip::Opening2D(&input, &kernel, &output5, x, y, 3,
                 dip::BOUNDARY_OPTION::REPLICATE_PADDING, 0.0);
  // Define a cv::Mat with the output of Opening2D.
  Mat outputImageConstantPaddingopening(sizesOutput[0], sizesOutput[1],
                                        CV_32FC1, output5.getData());
  imwrite(argv[6], outputImageConstantPaddingopening);
  Mat o5 = imread(argv[6], IMREAD_GRAYSCALE);
  // Define a cv::mat to store the output of Opencv's opening.
  Mat opencvConstantPaddingopening;
  morphologyEx(image, opencvConstantPaddingopening, 2, kernel1, cv::Point(x, y),
               3, cv::BORDER_REPLICATE, 0.0);
  if (!testImages(o5, opencvConstantPaddingopening)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }

  dip::Opening2D(&input, &kernel, &output6, x, y, 2,
                 dip::BOUNDARY_OPTION::CONSTANT_PADDING, 0);
  Mat outputImageReplicatePaddingopening(sizesOutput[0], sizesOutput[1],
                                         CV_32FC1, output6.getData());
  imwrite(argv[7], outputImageReplicatePaddingopening);

  Mat o6 = imread(argv[7], IMREAD_GRAYSCALE);
  // cv::Mat to store output of the opening op
  Mat opencvReplicatePaddingopening;
  morphologyEx(image, opencvReplicatePaddingopening, 2, kernel1,
               cv::Point(x, y), 2, 0, 0);

  if (!testImages(o6, opencvReplicatePaddingopening)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }

  dip::Closing2D(&input, &kernel, &output7, x, y, 3,
                 dip::BOUNDARY_OPTION::REPLICATE_PADDING, 0);
  Mat outputImageReplicatePaddingclosing(sizesOutput[0], sizesOutput[1],
                                         CV_32FC1, output7.getData());
  imwrite(argv[8], outputImageReplicatePaddingclosing);

  Mat o7 = imread(argv[8], IMREAD_GRAYSCALE);
  // cv::Mat to store output of the permutations of closing op
  Mat opencvReplicatePaddingclosing;
  morphologyEx(image, opencvReplicatePaddingclosing, 3, kernel1,
               cv::Point(x, y), 3, cv::BORDER_REPLICATE, 0);

  if (!testImages(o7, opencvReplicatePaddingclosing)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }

  dip::Closing2D(&input, &kernel, &output8, x, y, 2,
                 dip::BOUNDARY_OPTION::CONSTANT_PADDING, 0.0);

  // Define a cv::Mat with the output of Closing2D.
  Mat outputImageConstantPaddingclosing(sizesOutput[0], sizesOutput[1],
                                        CV_32FC1, output8.getData());
  imwrite(argv[9], outputImageConstantPaddingclosing);

  Mat o8 = imread(argv[9], IMREAD_GRAYSCALE);
  Mat opencvConstantPaddingclosing;
  morphologyEx(image, opencvConstantPaddingclosing, 3, kernel1, cv::Point(x, y),
               1, 0, 0.0);

  if (!testImages(o8, opencvConstantPaddingclosing)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }

  dip::TopHat2D(&input, &kernel, &output9, x, y, 3,
                dip::BOUNDARY_OPTION::REPLICATE_PADDING, 0);
  Mat outputImageReplicatePaddingtophat(sizesOutput[0], sizesOutput[1],
                                        CV_32FC1, output9.getData());
  imwrite(argv[10], outputImageReplicatePaddingtophat);

  Mat o9 = imread(argv[10], IMREAD_GRAYSCALE);
  // cv::Mat to store output of the permutations of tophat op
  Mat opencvReplicatePaddingtophat;
  morphologyEx(image, opencvReplicatePaddingtophat, 5, kernel1, cv::Point(x, y),
               3, cv::BORDER_REPLICATE, 0);

  if (!testImages(o9, opencvReplicatePaddingtophat)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }

  dip::TopHat2D(&input, &kernel, &output10, x, y, 2,
                dip::BOUNDARY_OPTION::CONSTANT_PADDING, 0.0);

  // Define a cv::Mat with the output of TopHat2D.
  Mat outputImageConstantPaddingtophat(sizesOutput[0], sizesOutput[1], CV_32FC1,
                                       output10.getData());
  imwrite(argv[11], outputImageConstantPaddingtophat);

  Mat o10 = imread(argv[11], IMREAD_GRAYSCALE);
  Mat opencvConstantPaddingtophat;
  morphologyEx(image, opencvConstantPaddingtophat, 5, kernel1, cv::Point(x, y),
               2, 0, 0.0);

  if (!testImages(o10, opencvConstantPaddingtophat)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }

  dip::BottomHat2D(&input, &kernel, &output11, x, y, 2,
                   dip::BOUNDARY_OPTION::REPLICATE_PADDING, 0);
  Mat outputImageReplicatePaddingbottomhat(sizesOutput[0], sizesOutput[1],
                                           CV_32FC1, output11.getData());
  imwrite(argv[12], outputImageReplicatePaddingbottomhat);

  Mat o11 = imread(argv[12], IMREAD_GRAYSCALE);
  // cv::Mat to store output of the permutations of bottomhat op
  Mat opencvReplicatePaddingbottomhat;
  morphologyEx(image, opencvReplicatePaddingbottomhat, 6, kernel1,
               cv::Point(x, y), 2, 1, 0);

  if (!testImages(o11, opencvReplicatePaddingbottomhat)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }

  dip::BottomHat2D(&input, &kernel, &output12, x, y, 2,
                   dip::BOUNDARY_OPTION::CONSTANT_PADDING, 0.0);

  // Define a cv::Mat with the output of BottomHat2D.
  Mat outputImageConstantPaddingbottomhat(sizesOutput[0], sizesOutput[1],
                                          CV_32FC1, output12.getData());
  imwrite(argv[13], outputImageConstantPaddingbottomhat);

  Mat o12 = imread(argv[13], IMREAD_GRAYSCALE);
  Mat opencvConstantPaddingbottomhat;
  morphologyEx(image, opencvConstantPaddingbottomhat, 6, kernel1,
               cv::Point(x, y), 2, 0, 0.0);

  if (!testImages(o12, opencvConstantPaddingbottomhat)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }

  dip::MorphGrad2D(&input, &kernel, &output13, x, y, 1,
                   dip::BOUNDARY_OPTION::CONSTANT_PADDING, 0.0);

  // Define a cv::Mat with the output of MorphGrad2D.
  Mat outputImageConstantPaddingmorphgrad(sizesOutput[0], sizesOutput[1],
                                          CV_32FC1, output13.getData());
  imwrite(argv[2], outputImageConstantPaddingmorphgrad);

  Mat o13 = imread(argv[2], IMREAD_GRAYSCALE);
  Mat opencvConstantPaddingmorphgrad;
  morphologyEx(image, opencvConstantPaddingmorphgrad, 4, kernel1,
               cv::Point(x, y), 1, 0, 0.0);

  if (!testImages(o13, opencvConstantPaddingmorphgrad)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }

  dip::MorphGrad2D(&input, &kernel, &output14, x, y, 2,
                   dip::BOUNDARY_OPTION::REPLICATE_PADDING, 0.0);

  // Define a cv::Mat with the output of MorphGrad2D.
  Mat outputImagereplicatePaddingmorphgrad(sizesOutput[0], sizesOutput[1],
                                           CV_32FC1, output14.getData());
  imwrite(argv[13], outputImageConstantPaddingmorphgrad);

  Mat o14 = imread(argv[13], IMREAD_GRAYSCALE);
  Mat opencvReplicatePaddingmorphgrad;
  morphologyEx(image, opencvReplicatePaddingmorphgrad, 4, kernel1,
               cv::Point(x, y), 2, 0, 0.0);

  if (!testImages(o14, opencvReplicatePaddingmorphgrad)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }

  return 1;
}

int main(int argc, char *argv[]) {
  std::ptrdiff_t x = 1;
  std::ptrdiff_t y = 1;
  testImplementation(argc, argv, x, y, 0);
  return 0;
}
