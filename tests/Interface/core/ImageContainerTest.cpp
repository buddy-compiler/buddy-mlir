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

bool compare_flt(float a, float b) { return (std::abs(a - b) < FLT_EPSILON); }

template <typename T, size_t N>
bool testImgcvnorm(cv::Mat testImgcv, Img<T, N> testImg, bool norm = false,
                   intptr_t sizes[N] = nullptr) {
  int cvn = testImgcv.dims;
  if (cvn != N)
    return false;
  for (size_t i = 0; i < N; ++i) {
    if (testImgcv.size[i] != testImg.getSizes()[i])
      return false;
  }
  T *data = testImg.getData();
  if (N == 2) {
    size_t k = 0;
    for (int i = 0; i < testImg.getSizes()[0]; ++i) {
      for (int j = 0; j < testImg.getSizes()[1]; ++j) {
        if (norm ? !compare_flt(data[k], (T)testImgcv.at<T>(i, j))
                 : !compare_flt(data[k], (T)testImgcv.at<uchar>(i, j)))
          return false;

        ++k;
      }
    }
    return true;
  } else if (N == 4) {
    if (sizes == nullptr) {
      return false;
    }
    size_t k = 0;
    // NCHW layout
    for (size_t batch = 0; batch < sizes[0]; ++batch) {
      for (size_t channel = 0; channel < sizes[1]; ++channel) {
        T *chandata = testImgcv.ptr<T>(batch, channel);
        for (size_t row = 0; row < sizes[2]; ++row) {
          for (size_t col = 0; col < sizes[3]; ++col) {
            if (!compare_flt(data[k], chandata[row * sizes[3] + col]))
              return false;

            ++k;
          }
        }
      }
    }
    return true;
  }
}

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
  Img<float, 2> testCopyConstructor3 = Img<float, 2>(grayimage_bmp);
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testCopyConstructor3[0]);
  Img<float, 2> *testCopyConstructor4 = new Img<float, 2>(grayimage_bmp);
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
  // Test Opencv Image without norm
  //===--------------------------------------------------------------------===//
  cv::Mat testImgcvbmp =
      cv::imread("../../../../tests/Interface/core/TestGrayImage.bmp",
                 cv::IMREAD_GRAYSCALE);
  Img<float, 2> testImgbmp(testImgcvbmp);
  bool testbmp = testImgcvnorm<float, 2>(testImgcvbmp, testImgbmp);
  // CHECK: 1
  fprintf(stderr, "%d \n", testbmp);
  //===--------------------------------------------------------------------===//
  // Test Opencv Image with norm
  //===--------------------------------------------------------------------===//
  Img<float, 2> testImgbmpnorm(testImgcvbmp, nullptr, true);
  cv::Mat checkimgbmp(testImgcvbmp.rows, testImgcvbmp.cols, CV_32FC1);
  testImgcvbmp.convertTo(checkimgbmp, CV_32FC1, 1.f / 255);
  bool testbmp1 = testImgcvnorm<float, 2>(checkimgbmp, testImgbmpnorm, true);
  // CHECK: 1
  fprintf(stderr, "%d \n", testbmp1);

  //===--------------------------------------------------------------------===//
  // Test Opencv blob Image (batched images) without norm (NCHW)
  //===--------------------------------------------------------------------===//
  std::vector<cv::Mat> testbmpvec = {testImgcvbmp, testImgcvbmp};
  cv::Mat testcvbmpblob = cv::dnn::blobFromImages(
      testbmpvec, 1.0, cv::Size(testImgcvbmp.rows, testImgcvbmp.cols));
  intptr_t sizesbmp[4] = {testcvbmpblob.size[0], testcvbmpblob.size[1],
                          testcvbmpblob.size[2], testcvbmpblob.size[3]};
  Img<float, 4> testImgbmpblob(testcvbmpblob, sizesbmp, false);
  bool testbmpN4 =
      testImgcvnorm<float, 4>(testcvbmpblob, testImgbmpblob, false, sizesbmp);
  // CHECK: 1
  fprintf(stderr, "%d \n", testbmpN4);

  //===--------------------------------------------------------------------===//
  // Test Opencv blob Image (batched images) with norm (NCHW)
  //===--------------------------------------------------------------------===//
  cv::Mat testcvbmpblob2 = cv::dnn::blobFromImages(
      testbmpvec, 1.0f / 255.0, cv::Size(testImgcvbmp.rows, testImgcvbmp.cols));
  Img<float, 4> testImgbmpblobnorm(testcvbmpblob, sizesbmp, true);
  bool testbmpN4norm = testImgcvnorm<float, 4>(
      testcvbmpblob2, testImgbmpblobnorm, true, sizesbmp);
  // CHECK: 1
  fprintf(stderr, "%d \n", testbmpN4norm);

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
  Img<float, 2> testCopyConstructor7 = Img<float, 2>(grayimage_jpg);
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testCopyConstructor7[0]);
  Img<float, 2> *testCopyConstructor8 = new Img<float, 2>(grayimage_jpg);
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
  // Test Opencv Image without norm
  //===--------------------------------------------------------------------===//
  cv::Mat testImgcvjpg =
      cv::imread("../../../../tests/Interface/core/TestGrayImage.jpg",
                 cv::IMREAD_GRAYSCALE);
  Img<float, 2> testImgjpg(testImgcvjpg);
  bool testjpg = testImgcvnorm<float, 2>(testImgcvjpg, testImgjpg);
  // CHECK: 1
  fprintf(stderr, "%d \n", testjpg);

  //===--------------------------------------------------------------------===//
  // Test Opencv Image with norm
  //===--------------------------------------------------------------------===//
  Img<float, 2> testImgjpgnorm(testImgcvjpg, nullptr, true);
  cv::Mat checkimgjpg(testImgcvjpg.rows, testImgcvjpg.cols, CV_32FC1);
  testImgcvjpg.convertTo(checkimgjpg, CV_32FC1, 1.f / 255);
  bool testjpg1 = testImgcvnorm<float, 2>(checkimgjpg, testImgjpgnorm, true);
  // CHECK: 1
  fprintf(stderr, "%d \n", testjpg1);

  //===--------------------------------------------------------------------===//
  // Test Opencv blob Image (batched images) without norm (NCHW)
  //===--------------------------------------------------------------------===//
  std::vector<cv::Mat> testjpgvec = {testImgcvjpg, testImgcvjpg};
  cv::Mat testcvjpgblob = cv::dnn::blobFromImages(
      testjpgvec, 1.0, cv::Size(testImgcvjpg.rows, testImgcvjpg.cols));
  intptr_t sizesjpg[4] = {testcvjpgblob.size[0], testcvjpgblob.size[1],
                          testcvjpgblob.size[2], testcvjpgblob.size[3]};
  Img<float, 4> testImgjpgblob(testcvjpgblob, sizesjpg, false);
  bool testjpgN4 =
      testImgcvnorm<float, 4>(testcvjpgblob, testImgjpgblob, false, sizesjpg);
  // CHECK: 1
  fprintf(stderr, "%d \n", testjpgN4);

  //===--------------------------------------------------------------------===//
  // Test Opencv blob Image (batched images) with norm (NCHW)
  //===--------------------------------------------------------------------===//
  cv::Mat testcvjpgblob2 = cv::dnn::blobFromImages(
      testjpgvec, 1.0f / 255.0, cv::Size(testImgcvjpg.rows, testImgcvjpg.cols));
  Img<float, 4> testImgjpgblobnorm(testcvjpgblob, sizesjpg, true);
  bool testjpgN4norm = testImgcvnorm<float, 4>(
      testcvjpgblob2, testImgjpgblobnorm, true, sizesjpg);
  // CHECK: 1
  fprintf(stderr, "%d \n", testjpgN4norm);

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
  Img<float, 2> testCopyConstructor11 = Img<float, 2>(grayimage_png);
  // CHECK: 15.0
  fprintf(stderr, "%f\n", testCopyConstructor11[0]);
  Img<float, 2> *testCopyConstructor12 = new Img<float, 2>(grayimage_png);
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

  //===--------------------------------------------------------------------===//
  // Test Opencv Image without norm
  //===--------------------------------------------------------------------===//
  cv::Mat testImgcvpng =
      cv::imread("../../../../tests/Interface/core/TestGrayImage.png",
                 cv::IMREAD_GRAYSCALE);
  Img<float, 2> testImgpng(testImgcvpng);
  bool testpng = testImgcvnorm<float, 2>(testImgcvpng, testImgpng);
  /// CHECK: 1
  fprintf(stderr, "%d \n", testpng);

  //===--------------------------------------------------------------------===//
  // Test Opencv Image with norm
  //===--------------------------------------------------------------------===//
  Img<float, 2> testImgpngnorm(testImgcvpng, nullptr, true);
  cv::Mat checkimgpng(testImgcvpng.rows, testImgcvpng.cols, CV_32FC1);
  testImgcvpng.convertTo(checkimgpng, CV_32FC1, 1.f / 255);
  bool testpng1 = testImgcvnorm<float, 2>(checkimgpng, testImgpngnorm, true);
  // CHECK: 1
  fprintf(stderr, "%d \n", testpng1);

  ///===--------------------------------------------------------------------===//
  // Test Opencv blob Image (batched images) without norm (NCHW)
  //===--------------------------------------------------------------------===//
  std::vector<cv::Mat> testpngvec = {testImgcvpng, testImgcvpng};
  cv::Mat testcvpngblob = cv::dnn::blobFromImages(
      testpngvec, 1.0, cv::Size(testImgcvpng.rows, testImgcvpng.cols));
  intptr_t sizespng[4] = {testcvpngblob.size[0], testcvpngblob.size[1],
                          testcvpngblob.size[2], testcvpngblob.size[3]};
  Img<float, 4> testImgpngblob(testcvpngblob, sizespng, false);
  bool testpngN4 =
      testImgcvnorm<float, 4>(testcvpngblob, testImgpngblob, false, sizespng);
  // CHECK: 1
  fprintf(stderr, "%d \n", testpngN4);

  //===--------------------------------------------------------------------===//
  // Test Opencv blob Image (batched images) with norm (NCHW)
  //===--------------------------------------------------------------------===//
  cv::Mat testcvpngblob2 = cv::dnn::blobFromImages(
      testpngvec, 1.0f / 255.0, cv::Size(testImgcvpng.rows, testImgcvpng.cols));
  Img<float, 4> testImgpngblobnorm(testcvpngblob, sizespng, true);
  bool testpngN4norm = testImgcvnorm<float, 4>(
      testcvpngblob2, testImgpngblobnorm, true, sizespng);
  // CHECK: 1
  fprintf(stderr, "%d \n", testpngN4norm);

  return 0;
}
