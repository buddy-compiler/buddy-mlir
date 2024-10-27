//===- NewImageContainerTestPng.cpp ---------------------------------------===//
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

// RUN: buddy-new-image-container-test-png 2>&1 | FileCheck %s

#include <buddy/DIP/ImgContainer.h>

int main() {
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
