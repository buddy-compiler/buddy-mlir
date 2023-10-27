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

#include "buddy/DIP/imgcodecs/loadsave.h"
#include <buddy/Core/Container.h>
#include <buddy/DIP/ImageContainer.h>

int main() {
  // The original test image is a gray scale image, and the pixel values are as
  // follows:
  // 15.0, 30.0, 45.0, 60.0
  // 75.0, 90.0, 105.0, 120.0
  // 135.0, 150.0, 165.0, 180.0
  // 195.0, 210.0, 225.0, 240.0
  // The test running directory is in <build dir>/tests/Interface/core, so the
  // `imread` function uses the following relative path.
  
  //===--------------------------------------------------------------------===//
  // Test bmp format image.
  //===--------------------------------------------------------------------===//
  Img<float, 2> grayimage_bmp = dip::imread<float, 2>(
      "../../../../tests/Interface/core/TestGrayImage.bmp",
      dip::IMGRD_GRAYSCALE);

  //===--------------------------------------------------------------------===//
  // Test copy constructor.
  //===--------------------------------------------------------------------===//
  Img<float, 2> testCopyConstructor1(grayimage_bmp);
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testCopyConstructor1[0]);
  // CHECK: 4, 4
  fprintf(stderr, "%ld, %ld\n", testCopyConstructor1.getSizes()[0],
          testCopyConstructor1.getSizes()[1]);
  // CHECK: 4, 1
  fprintf(stderr, "%ld, %ld\n", testCopyConstructor1.getStrides()[0],
          testCopyConstructor1.getStrides()[1]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", testCopyConstructor1.getRank());
  // CHECK: 16
  fprintf(stderr, "%ld\n", testCopyConstructor1.getSize());
  // CHECK: 60.0
  fprintf(stderr, "%f\n", testCopyConstructor1[3]);

  Img<float, 2> testCopyConstructor2 = grayimage_bmp;
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testCopyConstructor2[0]);
  // CHECK: 4, 4
  fprintf(stderr, "%ld, %ld\n", testCopyConstructor2.getSizes()[0],
          testCopyConstructor2.getSizes()[1]);
  // CHECK: 4, 1
  fprintf(stderr, "%ld, %ld\n", testCopyConstructor2.getStrides()[0],
          testCopyConstructor2.getStrides()[1]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", testCopyConstructor2.getRank());
  // CHECK: 16
  fprintf(stderr, "%ld\n", testCopyConstructor2.getSize());
  // CHECK: 60.0
  fprintf(stderr, "%f\n", testCopyConstructor2[3]);
  Img<float, 2> testCopyConstructor3 =
      Img<float, 2>(grayimage_bmp);
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testCopyConstructor3[0]);
  Img<float, 2> *testCopyConstructor4 =
      new Img<float, 2>(grayimage_bmp);
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testCopyConstructor4->getData()[0]);
  delete testCopyConstructor4;

  //===--------------------------------------------------------------------===//
  // Test move constructor.
  //===--------------------------------------------------------------------===//
  Img<float, 2> testMoveConstructor1(std::move(testCopyConstructor1));
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testMoveConstructor1[0]);
  // CHECK: 4, 4
  fprintf(stderr, "%ld, %ld\n", testMoveConstructor1.getSizes()[0],
          testMoveConstructor1.getSizes()[1]);
  // CHECK: 4, 1
  fprintf(stderr, "%ld, %ld\n", testMoveConstructor1.getStrides()[0],
          testMoveConstructor1.getStrides()[1]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", testMoveConstructor1.getRank());
  // CHECK: 16
  fprintf(stderr, "%ld\n", testMoveConstructor1.getSize());
  // CHECK: 60.0
  fprintf(stderr, "%f\n", testMoveConstructor1[3]);

  Img<float, 2> testMoveConstructor2 = std::move(testMoveConstructor1);
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testMoveConstructor2[0]);
  // CHECK: 4, 4
  fprintf(stderr, "%ld, %ld\n", testMoveConstructor2.getSizes()[0],
          testMoveConstructor2.getSizes()[1]);
  // CHECK: 4, 1
  fprintf(stderr, "%ld, %ld\n", testMoveConstructor2.getStrides()[0],
          testMoveConstructor2.getStrides()[1]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", testMoveConstructor2.getRank());
  // CHECK: 16
  fprintf(stderr, "%ld\n", testMoveConstructor2.getSize());
  // CHECK: 60.0
  fprintf(stderr, "%f\n", testMoveConstructor2[3]);

  //===--------------------------------------------------------------------===//
  // Test overloading bracket operator.
  //===--------------------------------------------------------------------===//
  Img<float, 2> testBracketOperator1(grayimage_bmp);
  // CHECK: 240.0
  fprintf(stderr, "%f\n", testBracketOperator1[15]);
  testBracketOperator1[15] = 90.0;
  // CHECK: 90.0
  fprintf(stderr, "%f\n", testBracketOperator1[15]);
  const Img<float, 2> testBracketOperator2(grayimage_bmp);
  // CHECK: 240.0
  fprintf(stderr, "%f\n", testBracketOperator2[15]);


  //===--------------------------------------------------------------------===//
  // Test jpeg format image.
  //===--------------------------------------------------------------------===//
  Img<float, 2> grayimage_jpg = dip::imread<float, 2>(
      "../../../../tests/Interface/core/TestGrayImage.jpg",
      dip::IMGRD_GRAYSCALE);

  //===--------------------------------------------------------------------===//
  // Test copy constructor.
  //===--------------------------------------------------------------------===//
  Img<float, 2> testCopyConstructor5(grayimage_jpg);
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testCopyConstructor5[0]);
  // CHECK: 4, 4
  fprintf(stderr, "%ld, %ld\n", testCopyConstructor5.getSizes()[0],
          testCopyConstructor5.getSizes()[1]);
  // CHECK: 4, 1
  fprintf(stderr, "%ld, %ld\n", testCopyConstructor5.getStrides()[0],
          testCopyConstructor5.getStrides()[1]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", testCopyConstructor5.getRank());
  // CHECK: 16
  fprintf(stderr, "%ld\n", testCopyConstructor5.getSize());
  // CHECK: 60.0
  fprintf(stderr, "%f\n", testCopyConstructor5[3]);

  Img<float, 2> testCopyConstructor6 = grayimage_jpg;
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testCopyConstructor6[0]);
  // CHECK: 4, 4
  fprintf(stderr, "%ld, %ld\n", testCopyConstructor6.getSizes()[0],
          testCopyConstructor6.getSizes()[1]);
  // CHECK: 4, 1
  fprintf(stderr, "%ld, %ld\n", testCopyConstructor6.getStrides()[0],
          testCopyConstructor6.getStrides()[1]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", testCopyConstructor6.getRank());
  // CHECK: 16
  fprintf(stderr, "%ld\n", testCopyConstructor6.getSize());
  // CHECK: 60.0
  fprintf(stderr, "%f\n", testCopyConstructor6[3]);
  Img<float, 2> testCopyConstructor7 =
      Img<float, 2>(grayimage_jpg);
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testCopyConstructor7[0]);
  Img<float, 2> *testCopyConstructor8 =
      new Img<float, 2>(grayimage_jpg);
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testCopyConstructor8->getData()[0]);
  delete testCopyConstructor8;

  //===--------------------------------------------------------------------===//
  // Test move constructor.
  //===--------------------------------------------------------------------===//
  Img<float, 2> testMoveConstructor3(std::move(testCopyConstructor5));
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testMoveConstructor3[0]);
  // CHECK: 4, 4
  fprintf(stderr, "%ld, %ld\n", testMoveConstructor3.getSizes()[0],
          testMoveConstructor3.getSizes()[1]);
  // CHECK: 4, 1
  fprintf(stderr, "%ld, %ld\n", testMoveConstructor3.getStrides()[0],
          testMoveConstructor3.getStrides()[1]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", testMoveConstructor3.getRank());
  // CHECK: 16
  fprintf(stderr, "%ld\n", testMoveConstructor3.getSize());
  // CHECK: 60.0
  fprintf(stderr, "%f\n", testMoveConstructor3[3]);

  Img<float, 2> testMoveConstructor4 = std::move(testMoveConstructor1);
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testMoveConstructor4[0]);
  // CHECK: 4, 4
  fprintf(stderr, "%ld, %ld\n", testMoveConstructor4.getSizes()[0],
          testMoveConstructor4.getSizes()[1]);
  // CHECK: 4, 1
  fprintf(stderr, "%ld, %ld\n", testMoveConstructor4.getStrides()[0],
          testMoveConstructor4.getStrides()[1]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", testMoveConstructor4.getRank());
  // CHECK: 16
  fprintf(stderr, "%ld\n", testMoveConstructor4.getSize());
  // CHECK: 60.0
  fprintf(stderr, "%f\n", testMoveConstructor4[3]);

  //===--------------------------------------------------------------------===//
  // Test overloading bracket operator.
  //===--------------------------------------------------------------------===//
  Img<float, 2> testBracketOperator3(grayimage_jpg);
  // CHECK: 240.0
  fprintf(stderr, "%f\n", testBracketOperator3[15]);
  testBracketOperator3[15] = 90.0;
  // CHECK: 90.0
  fprintf(stderr, "%f\n", testBracketOperator3[15]);
  const Img<float, 2> testBracketOperator4(grayimage_jpg);
  // CHECK: 240.0
  fprintf(stderr, "%f\n", testBracketOperator4[15]);


  //===--------------------------------------------------------------------===//
  // Test png format image.
  //===--------------------------------------------------------------------===//
  Img<float, 2> grayimage_png = dip::imread<float, 2>(
      "../../../../tests/Interface/core/TestGrayImage.png",
      dip::IMGRD_GRAYSCALE);

  //===--------------------------------------------------------------------===//
  // Test copy constructor.
  //===--------------------------------------------------------------------===//
  Img<float, 2> testCopyConstructor9(grayimage_png);
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testCopyConstructor9[0]);
  // CHECK: 4, 4
  fprintf(stderr, "%ld, %ld\n", testCopyConstructor9.getSizes()[0],
          testCopyConstructor9.getSizes()[1]);
  // CHECK: 4, 1
  fprintf(stderr, "%ld, %ld\n", testCopyConstructor9.getStrides()[0],
          testCopyConstructor9.getStrides()[1]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", testCopyConstructor9.getRank());
  // CHECK: 16
  fprintf(stderr, "%ld\n", testCopyConstructor9.getSize());
  // CHECK: 60.0
  fprintf(stderr, "%f\n", testCopyConstructor9[3]);

  Img<float, 2> testCopyConstructor10 = grayimage_png;
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testCopyConstructor10[0]);
  // CHECK: 4, 4
  fprintf(stderr, "%ld, %ld\n", testCopyConstructor10.getSizes()[0],
          testCopyConstructor10.getSizes()[1]);
  // CHECK: 4, 1
  fprintf(stderr, "%ld, %ld\n", testCopyConstructor10.getStrides()[0],
          testCopyConstructor10.getStrides()[1]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", testCopyConstructor10.getRank());
  // CHECK: 16
  fprintf(stderr, "%ld\n", testCopyConstructor10.getSize());
  // CHECK: 60.0
  fprintf(stderr, "%f\n", testCopyConstructor10[3]);
  Img<float, 2> testCopyConstructor11 =
      Img<float, 2>(grayimage_png);
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testCopyConstructor11[0]);
  Img<float, 2> *testCopyConstructor12 =
      new Img<float, 2>(grayimage_png);
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testCopyConstructor12->getData()[0]);
  delete testCopyConstructor12;

  //===--------------------------------------------------------------------===//
  // Test move constructor.
  //===--------------------------------------------------------------------===//
  Img<float, 2> testMoveConstructor5(std::move(testCopyConstructor9));
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testMoveConstructor5[0]);
  // CHECK: 4, 4
  fprintf(stderr, "%ld, %ld\n", testMoveConstructor5.getSizes()[0],
          testMoveConstructor5.getSizes()[1]);
  // CHECK: 4, 1
  fprintf(stderr, "%ld, %ld\n", testMoveConstructor5.getStrides()[0],
          testMoveConstructor5.getStrides()[1]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", testMoveConstructor5.getRank());
  // CHECK: 16
  fprintf(stderr, "%ld\n", testMoveConstructor5.getSize());
  // CHECK: 60.0
  fprintf(stderr, "%f\n", testMoveConstructor5[3]);

  Img<float, 2> testMoveConstructor6 = std::move(testMoveConstructor1);
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testMoveConstructor6[0]);
  // CHECK: 4, 4
  fprintf(stderr, "%ld, %ld\n", testMoveConstructor6.getSizes()[0],
          testMoveConstructor6.getSizes()[1]);
  // CHECK: 4, 1
  fprintf(stderr, "%ld, %ld\n", testMoveConstructor6.getStrides()[0],
          testMoveConstructor6.getStrides()[1]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", testMoveConstructor6.getRank());
  // CHECK: 16
  fprintf(stderr, "%ld\n", testMoveConstructor6.getSize());
  // CHECK: 60.0
  fprintf(stderr, "%f\n", testMoveConstructor6[3]);

  //===--------------------------------------------------------------------===//
  // Test overloading bracket operator.
  //===--------------------------------------------------------------------===//
  Img<float, 2> testBracketOperator5(grayimage_png);
  // CHECK: 240.0
  fprintf(stderr, "%f\n", testBracketOperator5[15]);
  testBracketOperator5[15] = 90.0;
  // CHECK: 90.0
  fprintf(stderr, "%f\n", testBracketOperator5[15]);
  const Img<float, 2> testBracketOperator6(grayimage_png);
  // CHECK: 240.0
  fprintf(stderr, "%f\n", testBracketOperator6[15]);
  
  return 0;
}