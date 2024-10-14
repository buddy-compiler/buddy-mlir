//===- NewImageContainerTest.cpp ------------------------------------------===//
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

// RUN: buddy-new-image-container-test 2>&1 | FileCheck %s

#include <buddy/DIP/ImgContainer.h>

int main() {
  //===--------------------------------------------------------------------===//
  // Test new image container - bmp format image.
  //===--------------------------------------------------------------------===//
  // Default Gray Scale
  dip::Image<float, 4> bmp32bitGrayDefault(
      "../../../../tests/Interface/core/TestImage-gray.bmp",
      dip::DIP_GRAYSCALE);
  // CHECK: BMP
  fprintf(stderr, "%s\n", bmp32bitGrayDefault.getFormatName().c_str());
  // CHECK: 28
  fprintf(stderr, "%ld\n", bmp32bitGrayDefault.getWidth());
  // CHECK: 28
  fprintf(stderr, "%ld\n", bmp32bitGrayDefault.getHeight());
  // CHECK: 32
  fprintf(stderr, "%d\n", bmp32bitGrayDefault.getBitDepth());
  // CHECK: 7
  fprintf(stderr, "%f\n", bmp32bitGrayDefault.getData()[0]);
  // Gray Scale + Normalization
  dip::Image<float, 4> bmp32bitGrayNorm(
      "../../../../tests/Interface/core/TestImage-gray.bmp", dip::DIP_GRAYSCALE,
      true /* norm */);
  // CHECK: BMP
  fprintf(stderr, "%s\n", bmp32bitGrayNorm.getFormatName().c_str());
  // CHECK: 28
  fprintf(stderr, "%ld\n", bmp32bitGrayNorm.getWidth());
  // CHECK: 28
  fprintf(stderr, "%ld\n", bmp32bitGrayNorm.getHeight());
  // CHECK: 32
  fprintf(stderr, "%d\n", bmp32bitGrayNorm.getBitDepth());
  // CHECK: 0.027451
  fprintf(stderr, "%f\n", bmp32bitGrayNorm.getData()[0]);

  // BMP 24bit Default Gray Scale
  dip::Image<float, 4> bmp24bitGrayDefault(
      "../../../../tests/Interface/core/TestImage-gray-24bit.bmp",
      dip::DIP_GRAYSCALE);
  // CHECK: BMP
  fprintf(stderr, "%s\n", bmp24bitGrayDefault.getFormatName().c_str());
  // CHECK: 28
  fprintf(stderr, "%ld\n", bmp24bitGrayDefault.getWidth());
  // CHECK: 28
  fprintf(stderr, "%ld\n", bmp24bitGrayDefault.getHeight());
  // CHECK: 24
  fprintf(stderr, "%d\n", bmp24bitGrayDefault.getBitDepth());
  // CHECK: 7
  fprintf(stderr, "%f\n", bmp24bitGrayDefault.getData()[0]);
  // BMP 24bit Gray Scale + Normalization
  dip::Image<float, 4> bmp24bitGrayNorm(
      "../../../../tests/Interface/core/TestImage-gray-24bit.bmp",
      dip::DIP_GRAYSCALE, true /* norm */);
  // CHECK: BMP
  fprintf(stderr, "%s\n", bmp24bitGrayNorm.getFormatName().c_str());
  // CHECK: 28
  fprintf(stderr, "%ld\n", bmp24bitGrayNorm.getWidth());
  // CHECK: 28
  fprintf(stderr, "%ld\n", bmp24bitGrayNorm.getHeight());
  // CHECK: 24
  fprintf(stderr, "%d\n", bmp24bitGrayNorm.getBitDepth());
  // CHECK: 0.027451
  fprintf(stderr, "%f\n", bmp24bitGrayNorm.getData()[0]);

  // BMP 16bit Default Gray Scale
  dip::Image<float, 4> bmp16bitGrayDefault(
      "../../../../tests/Interface/core/TestImage-gray-16bit-rgb565.bmp",
      dip::DIP_GRAYSCALE);
  // CHECK: BMP
  fprintf(stderr, "%s\n", bmp16bitGrayDefault.getFormatName().c_str());
  // CHECK: 28
  fprintf(stderr, "%ld\n", bmp16bitGrayDefault.getWidth());
  // CHECK: 28
  fprintf(stderr, "%ld\n", bmp16bitGrayDefault.getHeight());
  // CHECK: 16
  fprintf(stderr, "%d\n", bmp16bitGrayDefault.getBitDepth());
  // CHECK: 2
  fprintf(stderr, "%f\n", bmp16bitGrayDefault.getData()[0]);
  // BMP 16bit Gray Scale + Normalization
  dip::Image<float, 4> bmp16bitGrayNorm(
      "../../../../tests/Interface/core/TestImage-gray-16bit-rgb565.bmp",
      dip::DIP_GRAYSCALE, true /* norm */);
  // CHECK: BMP
  fprintf(stderr, "%s\n", bmp16bitGrayNorm.getFormatName().c_str());
  // CHECK: 28
  fprintf(stderr, "%ld\n", bmp16bitGrayNorm.getWidth());
  // CHECK: 28
  fprintf(stderr, "%ld\n", bmp16bitGrayNorm.getHeight());
  // CHECK: 16
  fprintf(stderr, "%d\n", bmp16bitGrayNorm.getBitDepth());
  // CHECK: 0.007843
  fprintf(stderr, "%f\n", bmp16bitGrayNorm.getData()[0]);

  dip::Image<float, 4> bmp32bitRGBDefault(
      "../../../../tests/Interface/core/TestImage-RGB-32bit.bmp", dip::DIP_RGB);
  // CHECK: BMP
  fprintf(stderr, "%s\n", bmp32bitRGBDefault.getFormatName().c_str());
  // CHECK: 224
  fprintf(stderr, "%ld\n", bmp32bitRGBDefault.getWidth());
  // CHECK: 224
  fprintf(stderr, "%ld\n", bmp32bitRGBDefault.getHeight());
  // CHECK: 32
  fprintf(stderr, "%d\n", bmp32bitRGBDefault.getBitDepth());
  // CHECK: 116
  fprintf(stderr, "%f\n", bmp32bitRGBDefault.getData()[0]);

  dip::Image<float, 4> bmp32bitRGBNorm(
      "../../../../tests/Interface/core/TestImage-RGB-32bit.bmp", dip::DIP_RGB,
      true);
  // CHECK: BMP
  fprintf(stderr, "%s\n", bmp32bitRGBNorm.getFormatName().c_str());
  // CHECK: 224
  fprintf(stderr, "%ld\n", bmp32bitRGBNorm.getWidth());
  // CHECK: 224
  fprintf(stderr, "%ld\n", bmp32bitRGBNorm.getHeight());
  // CHECK: 32
  fprintf(stderr, "%d\n", bmp32bitRGBNorm.getBitDepth());
  // CHECK: 0.45490
  fprintf(stderr, "%f\n", bmp32bitRGBNorm.getData()[0]);

  dip::Image<float, 4> bmp24bitRGBDefault(
      "../../../../tests/Interface/core/TestImage-RGB-24bit.bmp", dip::DIP_RGB);
  // CHECK: BMP
  fprintf(stderr, "%s\n", bmp24bitRGBDefault.getFormatName().c_str());
  // CHECK: 224
  fprintf(stderr, "%ld\n", bmp24bitRGBDefault.getWidth());
  // CHECK: 224
  fprintf(stderr, "%ld\n", bmp24bitRGBDefault.getHeight());
  // CHECK: 24
  fprintf(stderr, "%d\n", bmp24bitRGBDefault.getBitDepth());
  // CHECK: 116
  fprintf(stderr, "%f\n", bmp24bitRGBDefault.getData()[0]);

  dip::Image<float, 4> bmp24bitRGBNorm(
      "../../../../tests/Interface/core/TestImage-RGB-24bit.bmp", dip::DIP_RGB,
      true);
  // CHECK: BMP
  fprintf(stderr, "%s\n", bmp24bitRGBNorm.getFormatName().c_str());
  // CHECK: 224
  fprintf(stderr, "%ld\n", bmp24bitRGBNorm.getWidth());
  // CHECK: 224
  fprintf(stderr, "%ld\n", bmp24bitRGBNorm.getHeight());
  // CHECK: 24
  fprintf(stderr, "%d\n", bmp24bitRGBNorm.getBitDepth());
  // CHECK: 0.45490
  fprintf(stderr, "%f\n", bmp24bitRGBNorm.getData()[0]);

  // Default Gray Scale
  dip::Image<float, 4> pngGrayDefault(
      "../../../../tests/Interface/core/TestGrayImage.png", dip::DIP_GRAYSCALE);
  // CHECK: PNG
  fprintf(stderr, "%s\n", pngGrayDefault.getFormatName().c_str());
  // CHECK: 4
  fprintf(stderr, "%ld\n", pngGrayDefault.getWidth());
  // CHECK: 4
  fprintf(stderr, "%ld\n", pngGrayDefault.getHeight());
  // CHECK: 8
  fprintf(stderr, "%d\n", pngGrayDefault.getBitDepth());
  // CHECK: 15
  fprintf(stderr, "%f\n", pngGrayDefault.getData()[0]);
  // Gray Scale + Normalization
  dip::Image<float, 4> pngGrayNorm(
      "../../../../tests/Interface/core/TestGrayImage.png", dip::DIP_GRAYSCALE,
      true /* norm */);
  // CHECK: PNG
  fprintf(stderr, "%s\n", pngGrayNorm.getFormatName().c_str());
  // CHECK: 4
  fprintf(stderr, "%ld\n", pngGrayNorm.getWidth());
  // CHECK: 4
  fprintf(stderr, "%ld\n", pngGrayNorm.getHeight());
  // CHECK: 8
  fprintf(stderr, "%d\n", pngGrayNorm.getBitDepth());
  // CHECK: 0.058824
  fprintf(stderr, "%f\n", pngGrayNorm.getData()[0]);

  dip::Image<float, 4> pngRGBDefault(
      "../../../../tests/Interface/core/TestImage-RGB.png", dip::DIP_RGB);
  // CHECK: PNG
  fprintf(stderr, "%s\n", pngRGBDefault.getFormatName().c_str());
  // CHECK: 224
  fprintf(stderr, "%ld\n", pngRGBDefault.getWidth());
  // CHECK: 224
  fprintf(stderr, "%ld\n", pngRGBDefault.getHeight());
  // CHECK: 8
  fprintf(stderr, "%d\n", pngRGBDefault.getBitDepth());
  // CHECK: 144
  fprintf(stderr, "%f\n", pngRGBDefault.getData()[0]);

  dip::Image<float, 4> pngRGBNorm(
      "../../../../tests/Interface/core/TestImage-RGB.png", dip::DIP_RGB,
      true /* norm */);
  // CHECK: PNG
  fprintf(stderr, "%s\n", pngRGBNorm.getFormatName().c_str());
  // CHECK: 224
  fprintf(stderr, "%ld\n", pngRGBNorm.getWidth());
  // CHECK: 224
  fprintf(stderr, "%ld\n", pngRGBNorm.getHeight());
  // CHECK: 8
  fprintf(stderr, "%d\n", pngRGBNorm.getBitDepth());
  // CHECK: 0.5647
  fprintf(stderr, "%f\n", pngRGBNorm.getData()[0]);

  return 0;
}
