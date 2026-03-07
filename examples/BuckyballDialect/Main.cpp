//===- BuckyballMatmulTest.cpp -------------------------------------------===//
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
// Buckyball MatMul operation test.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Matrix size definitions
#define SIZE_M 1024
#define SIZE_N 1024
#define SIZE_K 1024

// Memory alignment macro
#define row_align(n) __attribute__((aligned(n)))

// MemRef structure definition
template <typename T, size_t N>
struct MemRef {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];

  MemRef(T *data, intptr_t sizes_[N]) {
    allocated = aligned = data;
    offset = 0;
    memcpy(sizes, sizes_, N * sizeof(intptr_t));
    strides[N - 1] = 1;
    for (int i = N - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * sizes[i + 1];
  }

  T *getData() { return aligned; }
};

// Declare MLIR generated function interface
extern "C" {
void _mlir_ciface_buckyball_matmul(MemRef<int8_t, 2> *input0,
                                   MemRef<int8_t, 2> *input1,
                                   MemRef<int8_t, 2> *output);
}

// Verification function
void verify(int8_t *expected, int8_t *actual, int m, int n, const char *name) {
  bool passed = true;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (expected[i * n + j] != actual[i * n + j]) {
        printf("Verification failed at [%d, %d]: expected %d, got %d\n",
               i, j, expected[i * n + j], actual[i * n + j]);
        passed = false;
        break;
      }
    }
    if (!passed)
      break;
  }
  
  if (passed)
    printf("%s verification: \033[32mPASSED\033[0m\n", name);
  else
    printf("%s verification: \033[31mFAILED\033[0m\n", name);
}

// Matrix printing function (for debugging)
void printMatrix(int8_t *matrix, int m, int n, const char *name) {
  printf("%s (%dx%d):\n", name, m, n);
  for (int i = 0; i < m && i < 10; i++) {
    for (int j = 0; j < n && j < 10; j++) {
      printf("%4d ", matrix[i * n + j]);
    }
    printf("%s\n", n > 10 ? "..." : "");
  }
  if (m > 10) printf("...\n");
  printf("\n");
}

// Simple matrix multiplication, used to generate expected results
void simpleMatmul(int8_t *a, int8_t *b, int8_t *c, int m, int n, int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      int32_t sum = 0;
      for (int p = 0; p < k; p++) {
        sum += a[i * k + p] * b[p * n + j];
      }
      // Handle potential overflow
      c[i * n + j] = (int8_t)(sum & 0xFF);
    }
  }
}

int main() {
  // Allocate matrix memory
  static int8_t inputA[SIZE_M * SIZE_K] row_align(1);
  static int8_t inputB[SIZE_K * SIZE_N] row_align(1);
  static int8_t outputC[SIZE_M * SIZE_N] row_align(1);
  static int8_t expectedC[SIZE_M * SIZE_N] row_align(1);
  
  // Initialize matrices - fill all with 1
  for (int i = 0; i < SIZE_M * SIZE_K; i++) {
    inputA[i] = 1;
  }
  
  for (int i = 0; i < SIZE_K * SIZE_N; i++) {
    inputB[i] = 1;
  }
  
  memset(outputC, 0, SIZE_M * SIZE_N);
  memset(expectedC, 0, SIZE_M * SIZE_N);
  
  // Set up MemRefs
  intptr_t sizesA[2] = {SIZE_M, SIZE_K};
  intptr_t sizesB[2] = {SIZE_K, SIZE_N};
  intptr_t sizesC[2] = {SIZE_M, SIZE_N};
  
  MemRef<int8_t, 2> inputAMemRef(inputA, sizesA);
  MemRef<int8_t, 2> inputBMemRef(inputB, sizesB);
  MemRef<int8_t, 2> outputCMemRef(outputC, sizesC);
  
  // Calculate expected result
  printf("Computing expected result...\n");
  simpleMatmul(inputA, inputB, expectedC, SIZE_M, SIZE_N, SIZE_K);
  
  // Call Buckyball matrix multiplication
  printf("Running Buckyball matrix multiplication...\n");
  _mlir_ciface_buckyball_matmul(&inputAMemRef, &inputBMemRef, &outputCMemRef);
  
  // Print partial results (optional)
  // printMatrix(expectedC, SIZE_M, SIZE_N, "Expected output");
  // printMatrix(outputC, SIZE_M, SIZE_N, "Actual output");
  
  // Verify results
  verify(expectedC, outputC, SIZE_M, SIZE_N, "Buckyball MatMul");
  
  return 0;
}
