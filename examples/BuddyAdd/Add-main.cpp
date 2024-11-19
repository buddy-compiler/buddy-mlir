//===- add-main.cpp -----------------------------------------------------===//
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
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace buddy;

constexpr size_t ParamsSize = 1;
constexpr size_t MaxVocabSize = 32000;
constexpr size_t MaxTokenLength = 40;
constexpr size_t HiddenSize = 4096;

/// Declare forward function.
extern "C" MemRef<float, 1>  _mlir_ciface_forward(MemRef<float, 1> , MemRef<float, 1> );

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

/// Capture input message.
void getUserInput(std::string &inputStr) {
  std::cout << "\nPlease input number:" << std::endl;
  std::cout << ">>> ";
  getline(std::cin, inputStr);
  std::cout << std::endl;
}

// Function to convert a string to a floating point number and fill it into the MemRef container.
void fillMemRefFromSingleFloatString(const std::string& str, MemRef<float, 1>& container) {
  float number = std::stof(str);
  std::fill(container.getData(), container.getData() + ParamsSize, number);
  std::cout << "The number is: " << container.getData()[0] << std::endl;
}

// -----------------------------------------------------------------------------
// Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  /// Print the title of this example.
  const std::string title = "\n HX's pretask Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  /// Get user message.
  std::string inputStr1, inputStr2;
  getUserInput(inputStr1);
  getUserInput(inputStr2);

  /// Initialize data containers
  //  - Input container.
  //  - Result container
  //  - Output container.
  //  - Parameters container.
  MemRef<float, 1> inputContainer1({ParamsSize});
  MemRef<float, 1> inputContainer2({ParamsSize});
  MemRef<float, 1> *resultContainer;
  MemRef<float, 1> resultContainer2({ParamsSize});

  /// Fill data into containers
  //  - Input:  inputStr1 inputStr2
  //  - Output: inputContainer1 + inputContainer2 to resultContainer
  fillMemRefFromSingleFloatString(inputStr1, inputContainer1);
  fillMemRefFromSingleFloatString(inputStr2, inputContainer2);
  
  resultContainer2 = _mlir_ciface_forward(inputContainer1, inputContainer2);
  std::cout << "size of resultContainer2: " << resultContainer2.getSize() << std::endl;

  /// Print the final result
  std::cout << "The result of " << inputStr1 << " + " << inputStr2 << " is: " << _mlir_ciface_forward(inputContainer1, inputContainer2).getData()[0] << std::endl;

  return 0;
}
