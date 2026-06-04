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
// WITHOUT WARRANTIES OR ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include <buddy/Core/Container.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// MLIR-generated subgraph.
// Current import/lowering keeps four memref arguments in the C wrapper, while
// the matmul body only consumes the latter x/weight pair.
extern "C" {
void _mlir_ciface_subgraph0(MemRef<float, 2> *result, MemRef<float, 2> *x,
                            MemRef<float, 2> *weight, MemRef<float, 2> *x_dup,
                            MemRef<float, 2> *weight_dup);
}

static void ref_matmul(const std::vector<float> &A, const std::vector<float> &B,
                       int64_t M, int64_t K, int64_t N,
                       std::vector<float> &out) {
  out.assign(static_cast<size_t>(M * N), 0.0f);
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int64_t p = 0; p < K; ++p) {
        acc += A[static_cast<size_t>(i * K + p)] *
               B[static_cast<size_t>(p * N + j)];
      }
      out[static_cast<size_t>(i * N + j)] = acc;
    }
  }
}

int main() {
  const int64_t M = 64;
  const int64_t K = 64;
  const int64_t N = 64;

  std::cout << "=== BuddyTorq (TORQ-Tile matmul) ===\n";

  std::vector<float> x_data(static_cast<size_t>(M * K), 1.0f);
  std::vector<float> w_data(static_cast<size_t>(K * N), 2.0f);

  intptr_t x_sizes[] = {M, K};
  intptr_t w_sizes[] = {K, N};
  MemRef<float, 2> x_desc(x_data.data(), x_sizes);
  MemRef<float, 2> w_desc(w_data.data(), w_sizes);

  intptr_t r_sizes[] = {M, N};
  MemRef<float, 2> result_desc(r_sizes, false, 0);

  _mlir_ciface_subgraph0(&result_desc, &x_desc, &w_desc, &x_desc, &w_desc);

  std::vector<float> ref;
  ref_matmul(x_data, w_data, M, K, N, ref);

  float *got = result_desc.getData();
  bool ok = true;
  float max_err = 0.0f;
  for (size_t i = 0; i < static_cast<size_t>(M * N); ++i) {
    float e = std::fabs(got[i] - ref[i]);
    max_err = std::max(max_err, e);
    if (e > 1e-2f)
      ok = false;
  }

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "max abs error vs reference: " << max_err << "\n";
  if (ok) {
    std::cout << "Test PASSED!\n";
    return 0;
  }
  std::cout << "Test FAILED!\n";
  return 1;
}
