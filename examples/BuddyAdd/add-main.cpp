//===- bert-main.cpp ------------------------------------------------------===//
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
_mlir_ciface_forward(MemRef<_Float16, 1> *result, 
                     MemRef<_Float16, 1> *arg0,
                     MemRef<_Float16, 1> *arg1);

std::ostream& PrintVec(_Float16 num, int dim) {
    std::cout << "[ ";
    for (int i = 0;i < dim; ++i) {
        std::cout << float(num) << ", ";
    }
    std::cout << "]";
    return std::cout;
}

int main() {
  /// Print the title of this example.
  const std::string title = "Add Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  /// Get user message and build Text container.
  std::cout << "What numbers do you want to add" << std::endl;
  std::cout << ">>> ";
  float num1,num2;
  std::cin >> num1 >> num2;
  _Float16 arg_0,arg_1;
  arg_0 = num1;
  arg_1 = num2;

  /// Initialize data containers.
  MemRef<_Float16, 1> result({10});
  MemRef<_Float16, 1> arg0({10},arg_0);
  MemRef<_Float16, 1> arg1({10},arg_1);

  PrintVec(arg_0,10) << " + ";
  PrintVec(arg_1,10) << " = ";
  
  const auto inferenceStart = std::chrono::high_resolution_clock::now();

  /// Execute forward inference of the model.
  _mlir_ciface_forward(&result, &arg0, &arg1);
  
  const auto inferenceEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;

  _Float16 init = result.getData()[0];
  for (int i = 0;i < 10; ++i) {
    if (init != result.getData()[i]) {
        std::cerr << "error in operator init is " << float(init) << " res is " << float(result.getData()[i]) << std::endl;
    }
  }
  assert(init == arg_0+arg_1);
  PrintVec(init,10) << std::endl;
  /// Print the performance.
  std::cout << "\033[33;1m[Time] \033[0m";
  std::cout << inferenceTime.count() << " ms"
            << std::endl;

  return 0;
}
