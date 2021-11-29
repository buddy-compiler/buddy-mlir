//====- edge-detection.cpp - Example of conv-opt tool ========================//
//
// This file implements an edge detection example with linalg.conv_2d operation.
// The linalg.conv_2d operation will be compiled into an object file with the
// conv-opt tool.
// This file will be linked with the object file to generate the executable
// file.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <time.h>

#include "kernels.h"

using namespace cv;
using namespace std;

// Define Memref Descriptor.
typedef struct MemRef_descriptor_ *MemRef_descriptor;
typedef struct MemRef_descriptor_ {
  float *allocated;
  float *aligned;
  intptr_t offset;
  intptr_t sizes[2];
  intptr_t strides[2];
} Memref;

// Constructor
MemRef_descriptor MemRef_Descriptor(float *allocated, float *aligned,
                                    intptr_t offset, intptr_t sizes[2],
                                    intptr_t strides[2]) {
  MemRef_descriptor n = (MemRef_descriptor)malloc(sizeof(*n));
  n->allocated = allocated;
  n->aligned = aligned;
  n->offset = offset;
  for (int i = 0; i < 2; i++)
    n->sizes[i] = sizes[i];
  for (int j = 0; j < 2; j++)
    n->strides[j] = strides[j];

  return n;
}

// Declare the conv2d C interface.
extern "C" {
void _mlir_ciface_conv_2d(MemRef_descriptor input, MemRef_descriptor kernel,
                          MemRef_descriptor output);
}

int main(int argc, char *argv[]) {
  printf("Start processing...\n");

  // Read as grayscale image.
  Mat image = imread(argv[1], IMREAD_GRAYSCALE);
  if (image.empty()) {
    cout << "Could not read the image: " << argv[1] << endl;
    return 1;
  }

  int inputSize = image.rows * image.cols;

  // Define the input with the image.
  float *inputAlign = (float *)malloc(inputSize * sizeof(float));
  int k = 0;
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      float pixelValue = (float)image.at<uchar>(i, j);
      inputAlign[k] = pixelValue;
      k++;
    }
  }

  // Define the kernel.
  // float kernelAlign[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
  float *kernelAlign = laplacianKernelAlign;
  int kernelRows = laplacianKernelRows;
  int kernelCols = laplacianKernelCols;

  // Define the output.
  int outputRows = image.rows - kernelRows + 1;
  int outputCols = image.cols - kernelCols + 1;
  float *outputAlign = (float *)malloc(outputRows * outputCols * sizeof(float));

  // Define the allocated, sizes, and strides.
  float *allocated = (float *)malloc(1 * sizeof(float));
  intptr_t sizesInput[2] = {image.rows, image.cols};
  intptr_t sizesKernel[2] = {kernelRows, kernelCols};
  intptr_t sizesOutput[2] = {outputRows, outputCols};
  intptr_t stridesInput[2] = {image.rows, image.cols};
  intptr_t stridesKernel[2] = {kernelRows, kernelCols};
  intptr_t stridesOutput[2] = {outputRows, outputCols};

  // Define memref descriptors.
  MemRef_descriptor input =
      MemRef_Descriptor(allocated, inputAlign, 0, sizesInput, stridesInput);
  MemRef_descriptor kernel =
      MemRef_Descriptor(allocated, kernelAlign, 0, sizesKernel, stridesKernel);
  MemRef_descriptor output =
      MemRef_Descriptor(allocated, outputAlign, 0, sizesOutput, stridesOutput);

  clock_t start,end;
  start = clock();

  // Call the MLIR conv2d function.
  _mlir_ciface_conv_2d(input, kernel, output);

  end = clock();
  cout << "Execution time: " 
       << (double)(end - start) / CLOCKS_PER_SEC << " s" << endl;

  // Define a cv::Mat with the output of the conv2d.
  Mat outputImage(outputRows, outputCols, CV_32FC1, output->aligned);

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

  free(inputAlign);
  free(outputAlign);
  free(input);
  free(kernel);
  free(output);

  return 0;
}
