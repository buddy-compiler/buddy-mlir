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
#include<buddy/DIP/imgcodecs/grfmt_bmp.hpp>
#include<buddy/DIP/imgcodecs/loadsave.hpp>
int main() {

  Img<uchar, 2> grayimage=imread<uchar,2>("../../../../tests/Interface/core/TestGrayImage_8.bmp", 1);
  
  Img<uchar, 2> testOpenCVConstructor(grayimage);
  // CHECK: 15
  fprintf(stderr, "%d\n", testOpenCVConstructor.getData()[0]);
  // CHECK: 4, 4
  fprintf(stderr, "%ld, %ld\n", testOpenCVConstructor._size[0],
          testOpenCVConstructor._size[1]);
  

}
