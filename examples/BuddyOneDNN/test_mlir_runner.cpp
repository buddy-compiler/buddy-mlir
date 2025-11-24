//===- test_mlir_runner.cpp -----------------------------------------------===//
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
// This file implements test runner for MLIR-generated code with oneDNN.
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// MLIR MemRef structure definition
template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

// MLIR-generated function declaration
extern "C" {
void _mlir_ciface_subgraph0(
    MemRefDescriptor<float, 3> *result, // Output [2, 4, 6]
    MemRefDescriptor<float, 3> *arg0,   // Input [2, 4, 8]
    MemRefDescriptor<float, 2> *arg1,   // Weight1 [8, 6] (unused)
    MemRefDescriptor<float, 2> *arg2,   // Weight2 [8, 6]
    MemRefDescriptor<float, 3> *arg3    // Bias [1, 1, 6]
);
}

// Helper function: Create MemRef
template <typename T, int N>
MemRefDescriptor<T, N> create_memref(std::vector<T> &data,
                                     const int64_t *sizes) {
  MemRefDescriptor<T, N> desc;
  desc.allocated = data.data();
  desc.aligned = data.data();
  desc.offset = 0;

  // Calculate strides (row-major)
  int64_t stride = 1;
  for (int i = N - 1; i >= 0; i--) {
    desc.sizes[i] = sizes[i];
    desc.strides[i] = stride;
    stride *= sizes[i];
  }

  return desc;
}

// Helper function: Print 3D tensor
void print_tensor_3d(const std::vector<float> &data, int64_t d0, int64_t d1,
                     int64_t d2) {
  std::cout << std::fixed << std::setprecision(2);
  for (int64_t i = 0; i < d0; i++) {
    std::cout << "Batch " << i << ":\n";
    for (int64_t j = 0; j < d1; j++) {
      std::cout << "  [";
      for (int64_t k = 0; k < d2; k++) {
        int64_t idx = i * d1 * d2 + j * d2 + k;
        std::cout << std::setw(8) << data[idx];
        if (k < d2 - 1)
          std::cout << ", ";
      }
      std::cout << "]\n";
    }
  }
}

int main() {
  std::cout << "=== MLIR + oneDNN Integration Test ===\n\n";

  // 1. Prepare input data
  // arg0: [2, 4, 8] - input tensor
  std::vector<float> arg0_data(2 * 4 * 8);
  for (size_t i = 0; i < arg0_data.size(); i++) {
    arg0_data[i] = 1.0f; // All ones
  }
  int64_t arg0_sizes[] = {2, 4, 8};
  auto arg0_desc = create_memref<float, 3>(arg0_data, arg0_sizes);

  // arg1: [8, 6] - weight1 (unused, but needs to be passed)
  std::vector<float> arg1_data(8 * 6, 0.0f);
  int64_t arg1_sizes[] = {8, 6};
  auto arg1_desc = create_memref<float, 2>(arg1_data, arg1_sizes);

  // arg2: [8, 6] - weight2 (for matmul)
  std::vector<float> arg2_data(8 * 6);
  for (size_t i = 0; i < arg2_data.size(); i++) {
    arg2_data[i] = 2.0f; // All twos
  }
  int64_t arg2_sizes[] = {8, 6};
  auto arg2_desc = create_memref<float, 2>(arg2_data, arg2_sizes);

  // arg3: [1, 1, 6] - bias
  std::vector<float> arg3_data(1 * 1 * 6);
  for (size_t i = 0; i < arg3_data.size(); i++) {
    arg3_data[i] = 0.5f; // All 0.5
  }
  int64_t arg3_sizes[] = {1, 1, 6};
  auto arg3_desc = create_memref<float, 3>(arg3_data, arg3_sizes);

  // 2. Prepare output buffer
  std::vector<float> result_data(2 * 4 * 6, 0.0f);
  int64_t result_sizes[] = {2, 4, 6};
  auto result_desc = create_memref<float, 3>(result_data, result_sizes);

  // 3. Call MLIR-generated function
  std::cout << "Running MLIR-generated subgraph0...\n";
  _mlir_ciface_subgraph0(&result_desc, &arg0_desc, &arg1_desc, &arg2_desc,
                         &arg3_desc);
  std::cout << "Done!\n\n";

  // 4. Read actual data from result_desc
  float *actual_result = result_desc.aligned;
  std::vector<float> actual_result_data(actual_result,
                                        actual_result + 2 * 4 * 6);

  // 5. Verify results
  // Expected value: reshape -> matmul(16.0) -> reshape -> add(+0.5) -> relu
  // = 16.5
  float expected = 16.5f;
  bool all_correct = true;
  int error_count = 0;

  for (size_t i = 0; i < actual_result_data.size(); i++) {
    if (std::abs(actual_result_data[i] - expected) > 1e-4) {
      all_correct = false;
      error_count++;
      if (error_count <= 5) { // Only print first 5 errors
        std::cerr << "Error at index " << i << ": expected " << expected
                  << ", got " << actual_result_data[i] << std::endl;
      }
    }
  }

  if (all_correct) {
    std::cout << " Test PASSED! All " << actual_result_data.size()
              << " values are correct (" << expected << ")\n";
    print_tensor_3d(actual_result_data, 2, 4, 6);
    return 0;
  } else {
    std::cout << " Test FAILED! " << error_count << " / "
              << actual_result_data.size() << " values are incorrect\n";
    std::cout << "\nFirst few results:\n";
    print_tensor_3d(actual_result_data, 2, 4, 6);
    return 1;
  }
}
