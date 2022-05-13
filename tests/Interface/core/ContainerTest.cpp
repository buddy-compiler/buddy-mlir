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
  //===--------------------------------------------------------------------===//
  // Test default shape constructor.
  //===--------------------------------------------------------------------===//
  MemRef<float, 2> testDefaultShapeConstructor(sizes);
  // CHECK: 0.0
  fprintf(stderr, "%f\n", testDefaultShapeConstructor.getData()[0]);
  // CHECK: 2, 3
  fprintf(stderr, "%ld, %ld\n", testDefaultShapeConstructor.getSizes()[0],
          testDefaultShapeConstructor.getSizes()[1]);
  // CHECK: 3, 1
  fprintf(stderr, "%ld, %ld\n", testDefaultShapeConstructor.getStrides()[0],
          testDefaultShapeConstructor.getStrides()[1]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", testDefaultShapeConstructor.getRank());
  // CHECK: 6
  fprintf(stderr, "%ld\n", testDefaultShapeConstructor.getSize());
  // CHECK: 0.0
  fprintf(stderr, "%f\n", testDefaultShapeConstructor[3]);

  //===--------------------------------------------------------------------===//
  // Test default shape constructor.
  //===--------------------------------------------------------------------===//
  MemRef<float, 2> testCustomShapeConstructor(sizes, 5);
  // CHECK: 5.0
  fprintf(stderr, "%f\n", testCustomShapeConstructor.getData()[0]);
  // CHECK: 5.0
  fprintf(stderr, "%f\n", testCustomShapeConstructor[5]);

  //===--------------------------------------------------------------------===//
  // Test overloading assignment operator.
  //===--------------------------------------------------------------------===//
  MemRef<float, 2> testConstructorAssingnment = testDefaultShapeConstructor;
  // CHECK: 0.0
  fprintf(stderr, "%f\n", testConstructorAssingnment[0]);
  testConstructorAssingnment = testCustomShapeConstructor;
  // CHECK: 5.0
  fprintf(stderr, "%f\n", testConstructorAssingnment[0]);
}

int main() {
  testShapeConstructor();
  return 0;
}
