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
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <type_traits>

using namespace buddy;

constexpr size_t ParamsSize = 6755192832;
constexpr size_t MaxVocabSize = 32000;
constexpr size_t MaxTokenLength = 40;
constexpr size_t HiddenSize = 4096;

/// Declare LLaMA forward function.
extern "C" void _mlir_ciface_forward(MemRef<float, 3> *, MemRef<float, 1> *,
                                     Text<size_t, 2> *);

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
  return std::distance(start, std::max_element(start, end));
}

// -----------------------------------------------------------------------------
// LLaMA Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  /// Print the title of this example.
  const std::string title = "LLaMA 2 Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

<<<<<<< HEAD
  // Initialize the container
  string pureStr;
  cout << "Please enter what you want to say to me" << endl;
  getline(cin, pureStr);
  auto buddyTokenizeStart = system_clock::now();
  Text<size_t, 2> pureStrContainer(pureStr);
  pureStrContainer.tokenizeLlama(vocabDir, 40);
  auto buddyTokenizeEnd = system_clock::now();
  auto buddyTokenizeTime =
      duration_cast<milliseconds>(buddyTokenizeEnd - buddyTokenizeStart);
  // Print the tokenized result
  cout << "Get User input:" << pureStrContainer.revertLlama(pureStrContainer)
       << endl;
  cout << "[Buddy] Tokenize input time: " << buddyTokenizeTime.count() << "ms"
       << endl;
  // Read the params
  auto buddyReadStart = system_clock::now();
  MemRef<float, 1> arg0({intptr_t(6755192832)});
  ifstream in0("../../examples/BuddyLlama/arg0.data", ios::in | ios::binary);
  std::cout << "use params file: "
            << std::filesystem::absolute("../../examples/BuddyLlama/arg0.data")
            << std::endl;
  if (!in0.is_open()) {
    throw std::runtime_error("Failed to open param file!");
  }
  in0.read((char *)(arg0.getData()), sizeof(float) * (arg0.getSize()));
  in0.close();
  auto buddyReadEnd = system_clock::now();
  auto buddyReadTime =
      duration_cast<milliseconds>(buddyReadEnd - buddyReadStart);
  cout << "Read params finish" << endl;
  cout << "[Buddy] Read params time: " << (double)(buddyReadTime.count()) / 1000
       << "s" << endl;
  // Run the model
  MemRef<float, 3> output[2] = {MemRef<float, 3>({1, 40, 32000}, false, 0),
                                MemRef<float, 3>({1, 40, 4096}, false, 0)};
  int generateLen = 40 - pureStrContainer.getTokenCnt();
  cout << "-----------------------start generate-----------------------"
       << endl;
  auto buddyStart = system_clock::now();
  for (int i = 0; i < generateLen; i++) {
    cout << "Iteration" << i << ": ";
    buddyReadStart = system_clock::now();
    // Perform calculations in memref generated by user input.
    _mlir_ciface_forward(output, &arg0, &pureStrContainer);
    int tokenIndex = pureStrContainer.getTokenCnt() - 1;
    int index = 0;
    float maxEle = output[0].getData()[tokenIndex * 32000];
    // Calculate the probability of occurrence of each token.
    for (int j = index + 1; j < 32000; j++) {
      if (output[0].getData()[tokenIndex * 32000 + j] > maxEle) {
        maxEle = output[0].getData()[tokenIndex * 32000 + j];
        index = j;
      }
    }
    pureStrContainer.getData()[pureStrContainer.getTokenCnt()] = index;
    // If the model generate 2(sep marker), interrupt generation immediately.
    if (index == 2) {
      free(output[0].release());
      free(output[1].release());
      break;
    }
    buddyReadEnd = system_clock::now();
    buddyReadTime = duration_cast<milliseconds>(buddyReadEnd - buddyReadStart);
    cout << pureStrContainer.getStr(index) << endl;
    cout << "[Buddy] Llama iteration " << i
         << " time: " << (double)(buddyReadTime.count()) / 1000 << "s" << endl;
    pureStrContainer.setTokenCnt(pureStrContainer.getTokenCnt() + 1);
    free(output[0].release());
    free(output[1].release());
=======
  /// Define directories of vacabulary and parameter file.
  const std::string vocabDir = "../../examples/BuddyLlama/vocab.txt";
  const std::string paramsDir = "../../examples/BuddyLlama/arg0.data";

  /// Get user message.
  std::string inputStr;
  getUserInput(inputStr);

  /// Initialize data containers
  //  - Input container.
  //  - Result container
  //  - Output container.
  //  - Parameters container.
  Text<size_t, 2> outputContainer;
  MemRef<float, 3> resultContainer[2] = {
      MemRef<float, 3>({1, MaxTokenLength, MaxVocabSize}, false, 0),
      MemRef<float, 3>({1, MaxTokenLength, HiddenSize}, false, 0)};
  Text<size_t, 2> inputContainer(inputStr);
  MemRef<float, 1> paramsContainer({ParamsSize});

  /// Fill data into containers
  //  - Input: register vocabulary and tokenize the input string.
  //  - Output: register vocabulary.
  //  - Parameters: load parameters from the `arg0` file into the container.
  tokenizeInput(vocabDir, inputContainer);
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
    const float *startPtr =
        resultContainer[0].getData() + tokenIndex * MaxVocabSize;
    const float *endPtr = startPtr + MaxVocabSize;
    int maxIndex = findMaxIndex(startPtr, endPtr);
    std::string tok = inputContainer.getStr(maxIndex);
    // Print the generated token and inference time.
    printIterInfo(i, tok, inferenceTime.count() / 1000);

    // Stop if a separator token (2, </s>) or line break token (13 <0x0A>) is
    // generated.
    if (maxIndex == 2) {
      break;
    }
    // Append the generated token into the input and output container.
    inputContainer.appendTokenIdx(maxIndex);
    outputContainer.appendTokenIdx(maxIndex);
    free(resultContainer[0].release());
    free(resultContainer[1].release());
>>>>>>> buddy-main
  }

  /// Print the final result
  std::cout << "\n\033[33;1m[Input]\033[0m " << inputStr << std::endl;
  std::cout << "\033[33;1m[Output]\033[0m " << outputContainer.revertLlama()
            << std::endl;

  return 0;
}
