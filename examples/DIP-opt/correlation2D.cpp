//====- edge-detection.cpp - Example of conv-opt tool ========================//
//
// This file implements an edge detection example with DIP.Corr2D operation.
// The DIP.Corr2D operation will be compiled into an object file with the
// DIP-opt tool.
// This file will be linked with the object file to generate the executable
// file.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/opencv.hpp>

#include <time.h>

#include "/home/prathamesh/buddy-mlir/examples/conv-opt/kernels.h"

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
void _mlir_ciface_DIPCorr2D(MemRef_descriptor input, MemRef_descriptor kernel,
                            MemRef_descriptor output, unsigned int centerX, unsigned int centerY, int boundaryOption);
}

bool testImages(cv::Mat img1, cv::Mat img2)
{
  if (img1.rows != img2.rows || img1.cols != img2.cols)
  {
    std::cout << "Dimensions not equal\n";
    return 0;
  }

  for (std::ptrdiff_t i = 0; i < img1.cols; ++i)
  {
    for (std::ptrdiff_t j = 0; j < img1.rows; ++j)
    {
      if (img1.at<uchar>(i, j) != img2.at<uchar>(i, j))
      {
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

bool testImplementation(int argc, char *argv[], 
    std::ptrdiff_t x, std::ptrdiff_t y, std::ptrdiff_t boundaryOption)
{
  // Read as grayscale image.
  Mat image = imread(argv[1], IMREAD_GRAYSCALE);
  if (image.empty()) {
    cout << "Could not read the image: " << argv[1] << endl;
  }

  int inputSize = image.rows * image.cols;

  // Define the input with the image.
  float *inputAlign = (float *)malloc(inputSize * sizeof(float));
  int k = 0;
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      float pixelValue = (float)image.at<uchar>(i, j);
      inputAlign[k] = pixelValue;
      // std::cout << "inputAlign Value   " << inputAlign[k] << "\n";
      k++;
    }
  }

  // Define the kernel.
  float kernelAlign[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  // float *kernelAlign = laplacianKernelAlign;
  int kernelRows = 3;
  int kernelCols = 3;

  // Define the output.
  int outputRows = image.rows;
  int outputCols = image.cols;
  float *outputAlign = (float *)malloc(outputRows * outputCols * sizeof(float));

  int k1 = 0;
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      outputAlign[k1] = 0;
      // std::cout << outputAlign[k1] << "  outputAlign\n";
      // std::cout << "inputAlign Value   " << inputAlign[k] << "\n";
      k1++;
    }
  }

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

  Mat kernel1 = Mat::ones(3, 3, CV_8UC1);

  // Call the MLIR Corr2D function.
  _mlir_ciface_DIPCorr2D(input, kernel, output, x, y, 0);

  // Define a cv::Mat with the output of the conv2d.
  Mat outputImage(outputRows, outputCols, CV_32FC1, output->aligned);

  // Choose a PNG compression level
  vector<int> compression_params;
  compression_params.push_back(IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);
  // std::cout << outputImage.cols << "  " << outputImage.rows << "\n";
  imwrite(argv[2], outputImage);

  Mat o1 = imread(argv[2], IMREAD_GRAYSCALE);
  Mat o2;
  filter2D(image, o2, CV_8UC1, kernel1, cv::Point(x, y), 0.0, cv::BORDER_CONSTANT);

  // std::cout << image << "\n\n";
  // std::cout << kernel1 << "\n\n";
  // std::cout << o1 << "\n\n";
  // std::cout << o2 << "\n\n\n\n";
  if (!testImages(o1, o2))
  {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }


  free(input);
  free(kernel);
  free(output);
  free(inputAlign);
  free(outputAlign);

  return 1;
}

int main(int argc, char *argv[]) {
  bool flag = 1;
  for (std::ptrdiff_t x = 0; x < 3; ++x)
  {
    for (std::ptrdiff_t y = 0; y < 3; ++y)
    {
      if (!testImplementation(argc, argv, x, y, 0))
      {
        flag = 0;
        break;
      }
      if (!flag)
      {
        break;
      }
    }
  }

  return 0;
}
