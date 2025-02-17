//===- buddy-deepseek-r1-main.cpp -----------------------------------------===//
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

constexpr size_t ParamsSize = 1777088064;
constexpr size_t MaxVocabSize = 151936;
constexpr size_t MaxTokenLength = 40;

/// Declare DeepSeekR1 forward function.
extern "C" void
_mlir_ciface_forward(MemRef<float, 3> *result, 
                      MemRef<float, 1> *arg0,
                      MemRef<long long, 2> *arg1,
                      MemRef<long long, 2> *arg2);

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

/// Capture input message.
void getUserInput(std::string &inputStr) {
  std::cout << "\nPlease send a message:" << std::endl;
  std::cout << ">>> ";
  getline(std::cin, inputStr);
  std::cout << std::endl;
}

/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

/// Print information for each iteration.
void printIterInfo(size_t iterIdx, int str, double time) {
  std::cout << "\033[32;1m[Iteration " << iterIdx << "] \033[0m";
  std::cout << "Token: " << str << " | "
            << "Time: " << time << "s" << std::endl;
}

/// Tokenize input data in the container.
void tokenizeInput(const std::string &vocabFile,
                   Text<size_t, 2> &inputContainer) {
  printLogLabel();
  std::cout << "Vocab file: " << std::filesystem::canonical(vocabFile)
            << std::endl;
  const auto buddyTokenizeStart = std::chrono::high_resolution_clock::now();
  inputContainer.tokenizeDeepSeekR1(vocabFile, MaxTokenLength);
  const auto buddyTokenizeEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> buddyTokenizeTime =
      buddyTokenizeEnd - buddyTokenizeStart;
  printLogLabel();
  std::cout << "Tokenize time: " << buddyTokenizeTime.count() << "ms"
            << std::endl;
}

/// Load parameters into data container.
void loadParameters(const std::string &paramFilePath,
                    MemRef<float, 1> &params) {
  const auto loadStart = std::chrono::high_resolution_clock::now();
  std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
  if (!paramFile.is_open()) {
    throw std::runtime_error("[Error] Failed to open params file!");
  }
  printLogLabel();
  std::cout << "Loading params..." << std::endl;
  printLogLabel();
  std::cout << "Params file: " << std::filesystem::canonical(paramFilePath)
            << std::endl;
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * (params.getSize()));
  if (paramFile.fail()) {
    throw std::runtime_error("Error occurred while reading params file!");
  }
  paramFile.close();
  const auto loadEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> loadTime =
      loadEnd - loadStart;
  printLogLabel();
  std::cout << "Params load time: " << (double)(loadTime.count()) / 1000
            << "s\n"
            << std::endl;
}

/// Find the index of the max value.
int findMaxIndex(const float *start, const float *end) {
  // std:: cout << "max element: " << *std::max_element(start, end) << std::endl; 
  return std::distance(start, std::max_element(start, end));
}

// -----------------------------------------------------------------------------
// DeepSeekR1 Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  /// Print the title of this example.
  const std::string title = "DeepSeekR1 Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  /// Define directories of vacabulary and parameter file.
  std::string deepSeekR1Dir = DEEPSEEKR1_EXAMPLE_PATH;
  std::string deepSeekR1BuildDir = DEEPSEEKR1_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = deepSeekR1Dir + "/vocab.txt";
  const std::string paramsDir = deepSeekR1BuildDir + "/arg0.data";

  /// Get user message.
  std::string inputStr;
  // getUserInput(inputStr);

  /// Initialize data containers
  //  - Input container.
  //  - Result container
  //  - Output container.
  //  - Parameters container.
  // Text<size_t, 2> outputContainer;
  MemRef<float, 3> resultContainer({1, 9, 151936});
  // Text<size_t, 2> input1Container(inputStr);
  MemRef<float, 1> paramsContainer({ParamsSize});
  MemRef<long long, 2> inputContainer({1, 40});
  MemRef<long long, 2> attention_mask({1, 40}, 0);
  MemRef<long long, 2> outputContainer({1, 40});
  long long data[] = {151646, 151646, 151644, 108386, 151645, 151648,    198};
  for (int i = 0; i < 7; i++) {
    inputContainer.getData()[i] = data[i];
    attention_mask.getData()[i] = 1;
  }

  /// Fill data into containers
  //  - Input: register vocabulary and tokenize the input string.
  //  - Output: register vocabulary.
  //  - Parameters: load parameters from the `arg0` file into the container.
  // tokenizeInput(vocabDir, input1Container);
  // for (int i = 0 ; i < 10 ; i ++ )
  //   std::cout << input1Container.getData()[i] << " ";
  // std::cout << std::endl;
  // outputContainer.loadVocab(vocabDir);
  loadParameters(paramsDir, paramsContainer);


  /// Run LLaMA Inference
  //  - Perform the forward function.
  //  - Find and append the generated token.
  //  - Continue iterating until the terminal condition is met.
  // int generateLen = MaxTokenLength - inputContainer.getTokenCnt();
  for (int i = 0; i < 33; i++) {
    const auto inferenceStart = std::chrono::high_resolution_clock::now();
    // std::cout << "input_ids:" << std::endl;
    // for (int j = 0 ; j < 40 ; j ++ )
    //   std::cout << inputContainer.getData()[j] << " ";
    // std::cout << std::endl;

    // std::cout << "attention_mask:" << std::endl;
    // for (int j = 0 ; j < 40 ; j ++ )
    //   std::cout << attention_mask.getData()[j] << " ";
    // std::cout << std::endl;

    // Execute the forward pass of the model.
    _mlir_ciface_forward(&resultContainer, &paramsContainer, &inputContainer, &attention_mask);

    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;

    // Determine the generated token.
    // int tokenIndex = inputContainer.getTokenCnt() - 1;
    int tokenIndex = 6 + i;
    const float *startPtr =
        resultContainer.getData() + tokenIndex * MaxVocabSize;
    const float *endPtr = startPtr + MaxVocabSize;
    int maxIndex = findMaxIndex(startPtr, endPtr);
    // std::string tok = inputContainer.getStr(maxIndex);
    // Print the generated token and inference time.
    // printIterInfo(i, tok, inferenceTime.count() / 1000);
    printIterInfo(i, maxIndex, inferenceTime.count() / 1000);

    // Stop if a separator token (2, </s>) or line break token (13 <0x0A>) is
    // generated.

    // Append the generated token into the input and output container.
    // inputContainer.appendTokenIdx(maxIndex);
    inputContainer.getData()[7 + i] = maxIndex;
    attention_mask.getData()[7 + i] = 1;
    outputContainer.getData()[7 + i] = maxIndex;
    // outputContainer.appendTokenIdx(maxIndex);
    free(resultContainer.release());

    if (maxIndex == 151643) {
      break;
    }
  }

  /// Print the final result
  std::cout << "\n\033[33;1m[Output]\033[0m " << "Result Token:" << std::endl;
  // std::cout << "\033[33;1m[Output]\033[0m " << outputContainer.revertLlama()
  //           << std::endl;



  for (int i = 7 ; i < 40 ; i ++ ){
    
    std::cout << outputContainer.getData()[i] << " ";
    if (outputContainer.getData()[i] == 151643)
      break;
  }

  std::cout << std::endl;

  return 0;
}
