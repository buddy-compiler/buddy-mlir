//===- ContainerTest.cpp --------------------------------------------------===//
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
// This is the MemRef container test file.
//
//===----------------------------------------------------------------------===//

// RUN: buddy-container-test 2>&1 | FileCheck %s

#include "Interface/buddy/core/Container.h"

void testShapeConstructor() {
  intptr_t sizes[] = {2, 3};
  MemRef<float, 2> testConstructor(sizes);
  // CHECK: 0.0
  fprintf(stderr, "%f\n", testConstructor.getData()[0]);
  // CHECK: 2, 3
  fprintf(stderr, "%ld, %ld\n", testConstructor.getSizes()[0],
          testConstructor.getSizes()[1]);
  // CHECK: 3, 1
  fprintf(stderr, "%ld, %ld\n", testConstructor.getStrides()[0],
          testConstructor.getStrides()[1]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", testConstructor.getRank());
  // CHECK: 6
  fprintf(stderr, "%ld\n", testConstructor.getSize());
  // CHECK: 0.0
  fprintf(stderr, "%f\n", testConstructor[3]);
}

int main() {
  testShapeConstructor();
  return 0;
}
