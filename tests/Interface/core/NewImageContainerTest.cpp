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
  dip::Image<float, 4> bmpGrayDefault(
      "../../../../tests/Interface/core/TestImage.bmp", dip::DIP_GRAYSCALE);
  // CHECK: BMP
  fprintf(stderr, "%s\n", bmpGrayDefault.getFormatName().c_str());
  // CHECK: 28
  fprintf(stderr, "%ld\n", bmpGrayDefault.getWidth());
  // CHECK: 28
  fprintf(stderr, "%ld\n", bmpGrayDefault.getHeight());
  // CHECK: 32
  fprintf(stderr, "%d\n", bmpGrayDefault.getBitDepth());
  // CHECK: 7
  fprintf(stderr, "%f\n", bmpGrayDefault.getData()[0]);
  // Gray Scale + Normalization
  dip::Image<float, 4> bmpGrayNorm(
      "../../../../tests/Interface/core/TestImage.bmp", dip::DIP_GRAYSCALE,
      true /* norm */);
  // CHECK: BMP
  fprintf(stderr, "%s\n", bmpGrayNorm.getFormatName().c_str());
  // CHECK: 28
  fprintf(stderr, "%ld\n", bmpGrayNorm.getWidth());
  // CHECK: 28
  fprintf(stderr, "%ld\n", bmpGrayNorm.getHeight());
  // CHECK: 32
  fprintf(stderr, "%d\n", bmpGrayNorm.getBitDepth());
  // CHECK: 0.027451
  fprintf(stderr, "%f\n", bmpGrayNorm.getData()[0]);

  return 0;
}
