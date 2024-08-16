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

extern "C" void
_mlir_ciface_forward(MemRef<float, 2> *result, MemRef<float, 4> *input);

int main() {
  /// Initialize data containers.
  MemRef<float, 4> input({1, 1, 28, 28});  
  MemRef<float, 2> result({28, 28});

  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      int index = i * 28 + j;
      input[index] = static_cast<float>(index);
    }
  }
  // Print the generated data to verify
  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      std::cout << input[i * 28 + j] << " ";
    }
    std::cout << std::endl;
  }

  const auto inferenceStart = std::chrono::high_resolution_clock::now();

  /// Execute forward inference of the model.
  _mlir_ciface_forward(&result, &input);
  
  const auto inferenceEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;

  /// Print the output data for verification.
  std::cout << "\033[33;1m[Output] \033[0m";
  std::cout << "[";
  for (int i = 0; i < 28; i++) {
    if (i > 0) std::cout << " ";
    std::cout << "[";
    for (int j = 0; j < 28; j++) {
      if (j > 0) std::cout << " ";
      std::cout << result[i * 28 + j];
    }
    std::cout << "]";
    if (i < 27) std::cout << "\n ";
  }
  std::cout << "]" << std::endl;

  /// Print the performance.
  std::cout << "\033[33;1m[Time] \033[0m";
  std::cout << inferenceTime.count() << " ms"
            << std::endl;

  return 0;
}
