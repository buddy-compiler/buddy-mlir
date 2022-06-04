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
  printf("Start processing...\n");

  // Read as grayscale image.
  Mat image = imread(argv[1], IMREAD_GRAYSCALE);
  if (image.empty()) {
    cout << "Could not read the image: " << argv[1] << endl;
    return 1;
  }
  Img<float, 2> input(image);

  // Define the kernel.
  float *kernelAlign = laplacianKernelAlign;
  int kernelRows = laplacianKernelRows;
  int kernelCols = laplacianKernelCols;
  intptr_t sizesKernel[2] = {kernelRows, kernelCols};
  MemRef<float, 2> kernel(kernelAlign, sizesKernel);

  // Define the output.
  int outputRows = image.rows - kernelRows + 1;
  int outputCols = image.cols - kernelCols + 1;
  intptr_t sizesOutput[2] = {outputRows, outputCols};
  MemRef<float, 2> output(sizesOutput);

  // Run the convolution and record the time.
  clock_t start, end;
  start = clock();

  // Call the MLIR conv2d function.
  _mlir_ciface_conv_2d(&input, &kernel, &output);

  end = clock();
  cout << "Execution time: " << (double)(end - start) / CLOCKS_PER_SEC << " s"
       << endl;

  // Define a cv::Mat with the output of the conv2d.
  Mat outputImage(outputRows, outputCols, CV_32FC1, output.getData());

  // Choose a PNG compression level
  vector<int> compression_params;
  compression_params.push_back(IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite(argv[2], outputImage, compression_params);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;

  return 0;
}
