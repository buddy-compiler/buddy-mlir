//===- ImageContainerTest.cpp ---------------------------------------------===//
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
// This is the image container test file.
//
//===----------------------------------------------------------------------===//

// RUN: buddy-image-container-test 2>&1 | FileCheck %s

#include <buddy/Core/Container.h>
#include <buddy/DIP/ImageContainer.h>
#include <opencv2/imgcodecs.hpp>

int main() {
  // The original test image is a gray scale image, and the pixel values are as
  // follows:
  // 15.0, 30.0, 45.0, 60.0
  // 75.0, 90.0, 105.0, 120.0
  // 135.0, 150.0, 165.0, 180.0
  // 195.0, 210.0, 225.0, 240.0
  // The test running directory is in <build dir>/tests/Interface/core, so the
  // `imread` function uses the following relative path.
  cv::Mat grayimage =
      cv::imread("../../../../tests/Interface/core/TestGrayImage.png",
                 cv::IMREAD_GRAYSCALE);
  //===--------------------------------------------------------------------===//
  // Test image constructor for OpenCV.
  //===--------------------------------------------------------------------===//
  Img<float, 2> testOpenCVConstructor(grayimage);
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testOpenCVConstructor.getData()[0]);
  // CHECK: 4, 4
  fprintf(stderr, "%ld, %ld\n", testOpenCVConstructor.getSizes()[0],
          testOpenCVConstructor.getSizes()[1]);
  // CHECK: 4, 1
  fprintf(stderr, "%ld, %ld\n", testOpenCVConstructor.getStrides()[0],
          testOpenCVConstructor.getStrides()[1]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", testOpenCVConstructor.getRank());
  // CHECK: 16
  fprintf(stderr, "%ld\n", testOpenCVConstructor.getSize());
  // CHECK: 60.0
  fprintf(stderr, "%f\n", testOpenCVConstructor[3]);

  //===--------------------------------------------------------------------===//
  // Test copy constructor.
  //===--------------------------------------------------------------------===//
  // TODO: Add copy assignment operator test.
  Img<float, 2> testCopyConstructor1(testOpenCVConstructor);
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testCopyConstructor1[0]);
  Img<float, 2> testCopyConstructor2 = testOpenCVConstructor;
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testCopyConstructor2[0]);
  Img<float, 2> testCopyConstructor3 = Img<float, 2>(testOpenCVConstructor);
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testCopyConstructor3[0]);
  Img<float, 2> *testCopyConstructor4 =
      new Img<float, 2>(testOpenCVConstructor);
  // CHECK: 5.0
  fprintf(stderr, "%f\n", testCopyConstructor4->getData()[0]);
  delete testCopyConstructor4;

  //===--------------------------------------------------------------------===//
  // Test move constructor.
  //===--------------------------------------------------------------------===//
  // TODO: Add copy assignment operator test.
  Img<float, 2> testMoveConstructor1(std::move(testCopyConstructor1));
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testMoveConstructor1[0]);
  Img<float, 2> testMoveConstructor2 = std::move(testMoveConstructor1);
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testMoveConstructor2[0]);

  //===--------------------------------------------------------------------===//
  // Test overloading bracket operator.
  //===--------------------------------------------------------------------===//
  Img<float, 2> testBracketOperator1(grayimage);
  // CHECK: 240.0
  fprintf(stderr, "%f\n", testBracketOperator1[15]);
  testBracketOperator1[15] = 90.0;
  // CHECK: 90.0
  fprintf(stderr, "%f\n", testBracketOperator1[15]);
  const Img<float, 2> testBracketOperator2(grayimage);
  // CHECK: 240.0
  fprintf(stderr, "%f\n", testBracketOperator2[15]);

  //===--------------------------------------------------------------------===//
  // Test image channels layout of RGB images from ImgContainer.
  //===--------------------------------------------------------------------===//
  // The test image is an RGB image with size 1026 * 1026 from
  // buddy-mlir/examples. The test running directory is in <build
  // dir>/tests/Interface/core, so the `imread` function uses the following
  // relative path.

  cv::Mat RGBimage = cv::imread("../../../../examples/ConvOpt/images/YuTu.png");

  // Represent NHWC layout by default.
  Img<float, 4> testRGBImageLayout1(RGBimage);
  // CHECK: 1, 1026, 1026, 3
  fprintf(stderr, "%ld, %ld, %ld, %ld\n", testRGBImageLayout1.getSizes()[0],
          testRGBImageLayout1.getSizes()[1], testRGBImageLayout1.getSizes()[2],
          testRGBImageLayout1.getSizes()[3]);

  // Represent NCHW layout with sizes = {1, 3, 1026, 1026}
  intptr_t sizesInput2[4] = {1, 3, RGBimage.rows, RGBimage.cols};
  Img<float, 4> testRGBImageLayout2(RGBimage, sizesInput2);
  // CHECK: 3158028, 1052676, 1026, 1
  fprintf(stderr, "%ld, %ld, %ld, %ld\n", testRGBImageLayout2.getStrides()[0],
          testRGBImageLayout2.getStrides()[1],
          testRGBImageLayout2.getStrides()[2],
          testRGBImageLayout2.getStrides()[3]);

  return 0;
}
