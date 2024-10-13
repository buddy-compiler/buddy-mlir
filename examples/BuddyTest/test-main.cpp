//===- test-main.cpp ------------------------------------------------------===//
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

#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace buddy;

// extern "C" void
// _mlir_ciface_forward(MemRef<float, 4> *result, MemRef<float, 4> *filter, MemRef<float, 1> *bias, MemRef<float, 4> *input);

extern "C" void
_mlir_ciface_forward(MemRef<float, 4> *result, MemRef<float, 4> *input);

int main() {
  /// Initialize data containers.
  const int N = 1;
  const int C = 1;
  const int K = 1;
  const int kernel_size = 2;
  const int stride = 2;
  const int H = 32;
  const int W = 32;
  const int H_out = H / kernel_size;
  const int W_out = W / kernel_size;

  MemRef<float, 4> input({N, C, H, W});  
  // MemRef<float, 4> filter({K, C, kernel_size, kernel_size});  
  // MemRef<float, 1> bias({K});  
  MemRef<float, 4> result({N, C, H_out, W_out});

  // Initial the input data
  for (int n = 0; n < N; n++) { 
    for (int c = 0; c < C; c++) {
      for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
          int index = n * C * H * W + c * H * W + i * W + j;
          input[index] = static_cast<float>((float)index/(H*W));
        }
      }
    }
  }
  // for (int k = 0; k < K; k++) { 
  //   for (int c = 0; c < C; c++) {
  //     for (int i = 0; i < kernel_size; i++) {
  //       for (int j = 0; j < kernel_size; j++) {
  //         int index = k * C * kernel_size * kernel_size + c * kernel_size * kernel_size + i * kernel_size + j;
  //         filter[index] = static_cast<float>(1);
  //       }
  //     }
  //   }
  // }
  
  // for (int k = 0; k < K; k++) {
  //   bias[k] = 1; 
  // }

  // Print the generated data to verify

  // for (int i = 0; i < H; i++) {
  //   for (int j = 0; j < W; j++) {
  //     std::cout << input[i * W + j] << " ";
  //   }
  //   std::cout << std::endl;
  // }

  const auto inferenceStart = std::chrono::high_resolution_clock::now();

  /// Execute forward inference of the model.
  _mlir_ciface_forward(&result, &input);
  
  const auto inferenceEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;

  /// Print the output data for verification.
  std::cout << "\033[33;1m[Output] \033[0m";
  std::cout << "[";
  for (int i = 0; i < H_out; i++) {
    if (i > 0) std::cout << " ";
    std::cout << "[";
    for (int j = 0; j < W_out; j++) {
      if (j > 0) std::cout << " ";
      std::cout << result[i * W_out + j];
    }
    std::cout << "]";
    if (i < H_out - 1) std::cout << "\n ";
  }
  std::cout << "]" << std::endl;

  /// Print the performance.
  std::cout << "\033[33;1m[Time] \033[0m";
  std::cout << inferenceTime.count() << " ms"
            << std::endl;

  return 0;
}
