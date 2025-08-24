//===- buddy-deepseek-r1-f16-main.cpp -------------------------------------===//
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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

using namespace buddy;

constexpr size_t ParamsSize = 1777088064;
constexpr size_t MaxVocabSize = 151936;
constexpr size_t MaxTokenLength = 40;

/// Declare DeepSeekR1 forward function.
extern "C" void _mlir_ciface_forward(MemRef<uint16_t, 3> *result,
                                     MemRef<uint16_t, 1> *arg0,
                                     Text<size_t, 2> *arg1,
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
void printIterInfo(size_t iterIdx, std::string str, double time) {
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
                    MemRef<uint16_t, 1> &params) {
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
                 sizeof(uint16_t) * (params.getSize()));
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

// f16 to f32 conversion function (IEEE 754 half precision -> single precision)
float decode_f16(uint16_t h) {
  uint16_t h_exp = (h & 0x7C00) >> 10;
  uint16_t h_sig = h & 0x03FF;
  uint16_t h_sign = h >> 15;

  if (h_exp == 0) {
    // subnormal
    float f = std::ldexp((float)h_sig, -24);
    return h_sign ? -f : f;
  } else if (h_exp == 0x1F) {
    // Inf/NaN
    return h_sig ? std::numeric_limits<float>::quiet_NaN()
                 : (h_sign ? -INFINITY : INFINITY);
  } else {
    float f = std::ldexp((float)(h_sig | 0x0400), h_exp - 25);
    return h_sign ? -f : f;
  }
}

int findMaxIndex(const uint16_t *start, size_t length) {
  int maxIdx = 0;
  float maxVal = decode_f16(start[0]);
  for (int i = 1; i < (int)length; ++i) {
    float val = decode_f16(start[i]);
    if (val > maxVal) {
      maxVal = val;
      maxIdx = i;
    }
  }
  return maxIdx;
}

// /// Find the index of the max value.
// int findMaxIndex(const uint16_t *start, const uint16_t *end) {
//   return std::distance(start, std::max_element(start, end));
// }

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
  const std::string paramsDir = deepSeekR1BuildDir + "/arg0-f16.data";

  /// Get user message.
  std::string inputStr;
  getUserInput(inputStr);

  /// Initialize data containers
  //  - Input container.
  //  - Result container
  //  - Output container.
  //  - Parameters container.
  Text<size_t, 2> outputContainer;
  MemRef<uint16_t, 3> resultContainer({1, 40, 151936});
  Text<size_t, 2> inputContainer(inputStr);
  MemRef<uint16_t, 1> paramsContainer({ParamsSize});
  MemRef<long long, 2> attention_mask({1, 40}, 0);

  /// Fill data into containers
  //  - Input: register vocabulary and tokenize the input string.
  //  - Output: register vocabulary.
  //  - Parameters: load parameters from the `arg0` file into the container.
  tokenizeInput(vocabDir, inputContainer);
  for (int i = 0; i < (int)inputContainer.getTokenCnt(); i++) {
    attention_mask.getData()[i] = 1;
  }
  outputContainer.loadVocab(vocabDir);
  loadParameters(paramsDir, paramsContainer);

  /// Run DeepSeekR1 Inference
  //  - Perform the forward function.
  //  - Find and append the generated token.
  //  - Continue iterating until the terminal condition is met.
  int generateLen = MaxTokenLength - inputContainer.getTokenCnt();
  for (int i = 0; i < generateLen; i++) {
    const auto inferenceStart = std::chrono::high_resolution_clock::now();
    // Execute the forward pass of the model.
    _mlir_ciface_forward(&resultContainer, &paramsContainer, &inputContainer,
                         &attention_mask);

    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;

    // Determine the generated token.
    int tokenIndex = inputContainer.getTokenCnt() - 1;
    const uint16_t *startPtr =
        resultContainer.getData() + tokenIndex * MaxVocabSize;
    // const uint16_t *endPtr = startPtr + MaxVocabSize;
    int maxIndex = findMaxIndex(startPtr, MaxVocabSize);
    std::string tok = inputContainer.getStr(maxIndex);
    // Print the generated token and inference time.
    printIterInfo(i, tok, inferenceTime.count() / 1000);

    // Stop if a <|end▁of▁sentence|> token is generated.
    if (maxIndex == 151643) {
      break;
    }
    // Append the generated token into the input and output container.
    inputContainer.appendTokenIdx(maxIndex);
    attention_mask.getData()[MaxTokenLength - generateLen + i] = 1;
    outputContainer.appendTokenIdx(maxIndex);
    free(resultContainer.release());
  }

  /// Print the final result
  std::cout << "\n\033[33;1m[Input]\033[0m " << inputStr << std::endl;
  std::cout << "\033[33;1m[Output]\033[0m "
            << outputContainer.revertDeepSeekR1() << std::endl;

  return 0;
}
