//====- edge-detection.cpp - Example of buddy-opt tool --------------------===//
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
// This file implements an edge detection example with linalg.conv_2d operation.
// The linalg.conv_2d operation will be compiled into an object file with the
// buddy-opt tool.
// This file will be linked with the object file to generate the executable
// file.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <time.h>
#include "Interface/buddy/core/Container.h"
#include "Interface/buddy/core/ImageContainer.h"
#include "kernels.h"

using namespace cv;
using namespace std;

// Declare the conv2d C interface.
extern "C" {
void _mlir_ciface_conv_2d(Img<float, 2> *input, MemRef<float, 2> *kernel,
                          MemRef<float, 2> *output);
}

int main(int argc, char *argv[]) {
  cout << "Start processing..." << endl;
  cout << "-----------------------------------------" << endl;
  //-------------------------------------------------------------------------//
  // Buddy Conv2D
  //-------------------------------------------------------------------------//

  /// Evaluate Buddy reading input process.
  clock_t buddyReadStart;
  buddyReadStart = clock();
  // Read as grayscale image.
  Mat buddyInputMat = imread(argv[1], IMREAD_GRAYSCALE);
  Img<float, 2> buddyInputMemRef(buddyInputMat);
  clock_t buddyReadEnd;
  buddyReadEnd = clock();
  double buddyReadTime =
      (double)(buddyReadEnd - buddyReadStart) / CLOCKS_PER_SEC;
  cout << "[Buddy] Read input time: " << buddyReadTime << " s" << endl;

  /// Evaluate Buddy defining kernel process.
  clock_t buddyKernelStart;
  buddyKernelStart = clock();
  // Get the data, row, and column information of the kernel.
  float *kernelAlign = laplacianKernelAlign;
  size_t kernelRows = laplacianKernelRows;
  size_t kernelCols = laplacianKernelCols;
  size_t sizesKernel[2] = {kernelRows, kernelCols};
  // Define the kernel MemRef object.
  MemRef<float, 2> kernelMemRef(kernelAlign, sizesKernel);
  clock_t buddyKernelEnd;
  buddyKernelEnd = clock();
  double buddyKernelTime =
      (double)(buddyKernelEnd - buddyKernelStart) / CLOCKS_PER_SEC;
  cout << "[Buddy] Define kernel time: " << buddyKernelTime << " s" << endl;

  /// Evaluate Buddy defining the output.
  clock_t buddyOutputStart;
  buddyOutputStart = clock();
  // Define the output.
  size_t outputRows = buddyInputMat.rows - kernelRows + 1;
  size_t outputCols = buddyInputMat.cols - kernelCols + 1;
  size_t sizesOutput[2] = {outputRows, outputCols};
  MemRef<float, 2> outputMemRef(sizesOutput);
  clock_t buddyOutputEnd;
  buddyOutputEnd = clock();
  double buddyOutputTime =
      (double)(buddyOutputEnd - buddyOutputStart) / CLOCKS_PER_SEC;
  cout << "[Buddy] Read output time: " << buddyOutputTime << " s" << endl;

  /// Evaluate the Buddy Conv2D.
  clock_t buddyConvStart;
  buddyConvStart = clock();
  // Perform the Conv2D function.
  _mlir_ciface_conv_2d(&buddyInputMemRef, &kernelMemRef, &outputMemRef);
  clock_t buddyConvEnd;
  buddyConvEnd = clock();
  double buddyConv2DTime =
      (double)(buddyConvEnd - buddyConvStart) / CLOCKS_PER_SEC;
  cout << "[Buddy] Perform Conv2D time: " << buddyConv2DTime << " s" << endl;

  /// Evaluate OpenCV writing output to image.
  clock_t buddyWriteStart;
  buddyWriteStart = clock();
  // Define a cv::Mat with the output of the Conv2D.
  Mat outputImage(outputRows, outputCols, CV_32FC1, outputMemRef.getData());
  // Choose a PNG compression level
  vector<int> buddyCompressionParams;
  buddyCompressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  buddyCompressionParams.push_back(9);
  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("buddy-result.png", outputImage, buddyCompressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  clock_t buddyWriteEnd;
  buddyWriteEnd = clock();
  double buddyWriteTime =
      (double)(buddyWriteEnd - buddyWriteStart) / CLOCKS_PER_SEC;
  cout << "[Buddy] Write image time: " << buddyWriteTime << " s" << endl;

  if (result) {
    cout << "[Buddy] Read + Conv2D time: "
         << buddyReadTime + buddyKernelTime + buddyOutputTime + buddyConv2DTime
         << endl;
    cout << "[Buddy] Total time: "
         << buddyReadTime + buddyKernelTime + buddyOutputTime +
                buddyConv2DTime + buddyWriteTime
         << endl;
  } else
    cout << "ERROR: Can't save PNG file." << endl;
  cout << "-----------------------------------------" << endl;

  //-------------------------------------------------------------------------//
  // OpenCV Filter2D
  //-------------------------------------------------------------------------//

  /// Evaluate OpenCV reading input process.
  clock_t ocvReadStart;
  ocvReadStart = clock();
  // Perform the read function
  Mat ocvInputImageFilter2D = imread(argv[1], IMREAD_GRAYSCALE);
  clock_t ocvReadEnd;
  ocvReadEnd = clock();
  double ocvReadTime = (double)(ocvReadEnd - ocvReadStart) / CLOCKS_PER_SEC;
  cout << "[OpenCV] Read input time: " << ocvReadTime << " s" << endl;

  /// Evaluate OpenCV defining kernel process.
  clock_t ocvKernelStart;
  ocvKernelStart = clock();
  // Get the data, row, and column information of the kernel.
  float *ocvKernelAlign = laplacianKernelAlign;
  int ocvKernelRows = laplacianKernelRows;
  int ocvKernelCols = laplacianKernelCols;
  // Define the kernel cv::Mat object.
  Mat ocvKernelFilter2D =
      Mat(ocvKernelRows, ocvKernelCols, CV_32FC1, ocvKernelAlign);
  clock_t ocvKernelEnd;
  ocvKernelEnd = clock();
  double ocvKernelTime =
      (double)(ocvKernelEnd - ocvKernelStart) / CLOCKS_PER_SEC;
  cout << "[OpenCV] Define kernel time: " << ocvKernelTime << " s" << endl;

  /// Evaluate OpenCV defining the output.
  clock_t ocvOutputStart;
  ocvOutputStart = clock();
  // Define a cv::Mat as the output object.
  Mat ocvOutputFilter2D;
  clock_t ocvOutputEnd;
  ocvOutputEnd = clock();
  double ocvOutputTime =
      (double)(ocvOutputEnd - ocvOutputStart) / CLOCKS_PER_SEC;
  cout << "[OpenCV] Define output time: " << ocvOutputTime << " s" << endl;

  /// Perform the OpenCV Filter2D.
  clock_t ocvFilter2DStart;
  ocvFilter2DStart = clock();
  // Perform the function with the input, output, and kernel cv::Mat object.
  filter2D(ocvInputImageFilter2D, ocvOutputFilter2D, CV_32FC1,
           ocvKernelFilter2D, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
  clock_t ocvFilter2DEnd;
  ocvFilter2DEnd = clock();
  double ocvFilter2DTime =
      (double)(ocvFilter2DEnd - ocvFilter2DStart) / CLOCKS_PER_SEC;
  cout << "[OpenCV] Filter2D time: " << ocvFilter2DTime << " s" << endl;

  /// Write the OpenCV output to image.
  clock_t ocvWriteStart;
  ocvWriteStart = clock();
  // Choose a PNG compression level
  vector<int> ocvCompressionParams;
  ocvCompressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  ocvCompressionParams.push_back(9);
  // Write output to PNG.
  bool ocvResult = false;
  try {
    ocvResult =
        imwrite("opencv-result.png", ocvOutputFilter2D, ocvCompressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  clock_t ocvWriteEnd;
  ocvWriteEnd = clock();
  double ocvWriteTime = (double)(ocvWriteEnd - ocvWriteStart) / CLOCKS_PER_SEC;
  cout << "[OpenCV] Write image time: " << ocvWriteTime << " s" << endl;

  if (ocvResult) {
    cout << "[OpenCV] Read + Filter2D time: "
         << ocvReadTime + ocvKernelTime + ocvOutputTime + ocvFilter2DTime
         << endl;
    cout << "[OpenCV] Total time: "
         << ocvReadTime + ocvKernelTime + ocvOutputTime + ocvFilter2DTime +
                ocvWriteTime
         << endl;
  } else
    cout << "ERROR: Can't save PNG file." << endl;
  cout << "-----------------------------------------" << endl;
  return 0;
}
