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

int main() {
  size_t sizes[] = {2, 3};
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
  // Test custom shape constructor.
  //===--------------------------------------------------------------------===//
  MemRef<float, 2> testCustomShapeConstructor(sizes, 5);
  // CHECK: 5.0
  fprintf(stderr, "%f\n", testCustomShapeConstructor.getData()[0]);
  // CHECK: 5.0
  fprintf(stderr, "%f\n", testCustomShapeConstructor[5]);

  //===--------------------------------------------------------------------===//
  // Test array constructor.
  //===--------------------------------------------------------------------===//
  float data[6] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  MemRef<float, 2> testArrayConstructor(data, sizes);
  // CHECK: 0.0
  fprintf(stderr, "%f\n", testArrayConstructor.getData()[0]);
  // CHECK: 5.0
  fprintf(stderr, "%f\n", testArrayConstructor[5]);

  //===--------------------------------------------------------------------===//
  // Test copy constructor and copy assignment operator.
  //===--------------------------------------------------------------------===//
  MemRef<float, 2> testCopyConstructor1(testCustomShapeConstructor);
  // CHECK: 5.0
  fprintf(stderr, "%f\n", testCopyConstructor1[0]);
  MemRef<float, 2> testCopyConstructor2 = testCustomShapeConstructor;
  // CHECK: 5.0
  fprintf(stderr, "%f\n", testCopyConstructor2[0]);
  MemRef<float, 2> testCopyConstructor3 =
      MemRef<float, 2>(testCustomShapeConstructor);
  // CHECK: 5.0
  fprintf(stderr, "%f\n", testCopyConstructor3[0]);
  MemRef<float, 2> *testCopyConstructor4 =
      new MemRef<float, 2>(testCustomShapeConstructor);
  // CHECK: 5.0
  fprintf(stderr, "%f\n", testCopyConstructor4->getData()[0]);
  delete testCopyConstructor4;
  MemRef<float, 2> testCopyAssingnment = testDefaultShapeConstructor;
  testCopyAssingnment = testCustomShapeConstructor;
  // CHECK: 5.0
  fprintf(stderr, "%f\n", testCopyAssingnment[0]);

  //===--------------------------------------------------------------------===//
  // Test move constructor and move assignment operator.
  //===--------------------------------------------------------------------===//
  MemRef<float, 2> tempMemRefContainer(testCustomShapeConstructor);
  MemRef<float, 2> testMoveConstructor1(std::move(tempMemRefContainer));
  // CHECK: 5.0
  fprintf(stderr, "%f\n", testMoveConstructor1[0]);
  MemRef<float, 2> testMoveConstructor2 = std::move(testMoveConstructor1);
  // CHECK: 5.0
  fprintf(stderr, "%f\n", testMoveConstructor2[0]);
  MemRef<float, 2> testMoveAssignment(testDefaultShapeConstructor);
  testMoveAssignment = std::move(testMoveConstructor2);
  // CHECK: 5.0
  fprintf(stderr, "%f\n", testMoveAssignment[0]);

  //===--------------------------------------------------------------------===//
  // Test overloading bracket operator.
  //===--------------------------------------------------------------------===//
  MemRef<float, 2> testBracketOperator1(sizes);
  // CHECK: 0.0
  fprintf(stderr, "%f\n", testBracketOperator1[0]);
  testBracketOperator1[0] = 5.0;
  // CHECK: 5.0
  fprintf(stderr, "%f\n", testBracketOperator1[0]);
  const MemRef<float, 2> testBracketOperator2(sizes);
  // CHECK: 0.0
  fprintf(stderr, "%f\n", testBracketOperator2[0]);

  return 0;
}
