//===- llama-main.cpp -----------------------------------------------------===//
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

// constexpr size_t ParamsSize = 6771970048;
// constexpr size_t MaxVocabSize = 32000;
// constexpr size_t MaxTokenLength = 40;
// constexpr size_t HiddenSize = 4096;
constexpr size_t ParamsSize = 108368;
constexpr size_t MaxVocabSize = 3000;
constexpr size_t MaxTokenLength = 40;
constexpr size_t HiddenSize = 16;


using fp16_t = uint16_t;
using bp16_t = uint16_t;

/// Declare LLaMA forward function.
extern "C" void _mlir_ciface_forward(MemRef<fp16_t, 3> *, MemRef<fp16_t, 1> *,
                                     Text<size_t, 2> *);

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

// ----------------- FP16/BP16 <-> F32 conversion utils & builtins ----------------------
extern "C" fp16_t __gnu_f2h_ieee(float f);
extern "C" float __gnu_h2f_ieee(fp16_t hf);
// extern "C" bp16_t __truncsfbf2(float f);

static fp16_t float2half(float f) {
  return __gnu_f2h_ieee(f);
}
static float half2float(fp16_t hf) {
   return __gnu_h2f_ieee(hf);
}

// static bp16_t float2bfloat(float floatValue) {
//   return __truncsfbf2(floatValue);
// }

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
  inputContainer.tokenizeLlama(vocabFile, MaxTokenLength);
  const auto buddyTokenizeEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> buddyTokenizeTime =
      buddyTokenizeEnd - buddyTokenizeStart;
  printLogLabel();
  std::cout << "Tokenize time: " << buddyTokenizeTime.count() << "ms"
            << std::endl;
}

/// Load parameters into data container.
void loadParameters(const std::string &paramFilePath,
                    MemRef<fp16_t, 1> &params) {
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
                 sizeof(fp16_t) * (params.getSize()));
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
  // for debug: output loaded param
  fp16_t *param_data = params.getData();
  std::cout << "[DEBUG] loaded params: " << std::endl;
  for (int i = 90; i < 100; i++) {
    std::cout << half2float(param_data[i]) << " ";
  }
  std::cout << std::endl;
}

/// Find the index of the max value.
int findMaxIndex(const fp16_t *start, const fp16_t *end) {
  return std::distance(start, std::max_element(start, end));
}

// -----------------------------------------------------------------------------
// LLaMA Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  /// Print the title of this example.
  const std::string title = "F16 LLaMA 2 Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  /// Define directories of vacabulary and parameter file.
  const std::string vocabDir = "../examples/BuddyF16Llama/vocab.txt";
  const std::string paramsDir = "../examples/BuddyF16Llama/fp16-params.data";

  /// Get user message.
  std::string inputStr;
  // getUserInput(inputStr);

  /// Initialize data containers
  //  - Input container.
  //  - Result container
  //  - Output container.
  //  - Parameters container.
  Text<size_t, 2> outputContainer;
  MemRef<fp16_t, 3> resultContainer[2] = {
      MemRef<fp16_t, 3>({1, MaxTokenLength, HiddenSize}, false, 0),
      MemRef<fp16_t, 3>({1, MaxTokenLength, MaxVocabSize}, false, 0)};
  Text<size_t, 2> inputContainer(inputStr);
  MemRef<fp16_t, 1> paramsContainer({ParamsSize});

  /// Fill data into containers
  //  - Input: register vocabulary and tokenize the input string.
  //  - Output: register vocabulary.
  //  - Parameters: load parameters from the `arg0` file into the container.
  tokenizeInput(vocabDir, inputContainer);
  // for debug
  inputContainer.setTokenCnt(0);
  assert (inputContainer.getTokenCnt() == 0);
  outputContainer.loadVocab(vocabDir);
  loadParameters(paramsDir, paramsContainer);

  /// Run LLaMA Inference
  //  - Perform the forward function.
  //  - Find and append the generated token.
  //  - Continue iterating until the terminal condition is met.
  int generateLen = MaxTokenLength - inputContainer.getTokenCnt();
  for (int i = 0; i < generateLen; i++) {
    const auto inferenceStart = std::chrono::high_resolution_clock::now();
    // Execute the forward pass of the model.
    _mlir_ciface_forward(resultContainer, &paramsContainer, &inputContainer);

    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;
    // Determine the generated token.
    int tokenIndex = inputContainer.getTokenCnt() - 1;
    const fp16_t *startPtr =
        resultContainer[1].getData() + tokenIndex * MaxVocabSize;
    const fp16_t *endPtr = startPtr + MaxVocabSize;
    // int maxIndex = findMaxIndex(startPtr, endPtr);
    int maxIndex = i;
    std::string tok = inputContainer.getStr(maxIndex);
    // Print the generated token and inference time.
    printIterInfo(i, tok, inferenceTime.count() / 1000);

    // for debug: print result
    std::cout << "[DEBUG] output: ";
    for (int i = 0; i < 10; i++) {
      std::cout << half2float(startPtr[i]) << " ";
    }
    std::cout << std::endl;

    // Stop if a separator token (2, </s>) or line break token (13 <0x0A>) is
    // generated.
    // if (maxIndex == 2) {
    //   break;
    // }
    // Append the generated token into the input and output container.
    inputContainer.appendTokenIdx(maxIndex);
    outputContainer.appendTokenIdx(maxIndex);
    free(resultContainer[0].release());
    free(resultContainer[1].release());
  }

  /// Print the final result
  std::cout << "\n\033[33;1m[Input]\033[0m " << inputStr << std::endl;
  std::cout << "\033[33;1m[Output]\033[0m " << outputContainer.revertLlama()
            << std::endl;

  return 0;
}
