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

#include "llama_utils.h"

using namespace buddy;

constexpr static bool debug = false;

/// Declare LLaMA forward function.
extern "C" void _mlir_ciface_forward(MemRef<float, 3> *,
                                     MemRef<float, 1> *,
                                     MemRef<int64_t, 1> *,
                                     MemRef<int8_t, 1> *,
                                     Text<size_t, 2> *);

/// Find the index of the max value.
int findMaxIndex(const float *start, const float *end) {
  return std::distance(start, std::max_element(start, end));
}

// -----------------------------------------------------------------------------
// LLaMA Inference Main Entry
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {

  /// Print the title of this example.
  const std::string title = "LLaMA 2 Inference Powered by Buddy Compiler with int8 quantization";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;
  if constexpr(debug) {
    std::cout << "\033[33;1m" << "Debug mode" << "\033[0m" << std::endl;
  }

  /// Define directories of vacabulary and parameter file.
  const std::string vocabDir = "../examples/BuddyLlama/vocab.txt";
  const std::string paramsDir = "../examples/BuddyLlama/";
  const std::string configDir = "../examples/BuddyLlama/config.txt";
  
  ModelConfig config;
  loadModelConfig(configDir, config);

  /// Initialize data containers
  //  - Input container.
  //  - Result container
  //  - Output container.
  //  - Parameters container.
  Text<size_t, 2> outputContainer;
  MemRef<float, 3> resultContainer[2] = {
      MemRef<float, 3>({1, config.maxTokenLength, config.hiddenSize}, false, 0),
      MemRef<float, 3>({1, config.maxTokenLength, config.maxVocabSize}, false, 0)};
  MemRef<float, 1> float32ParamsContainer({config.paramSize});
  MemRef<int64_t, 1> int64ParamsContainer({config.paramSize});
  MemRef<int8_t, 1> int8ParamsContainer({config.paramSize});
  MemRef<float, 1> expectedOutputContainer({config.maxTokenLength * config.maxVocabSize});

  outputContainer.loadVocab(vocabDir);
  loadParameters<float>(paramsDir + "params-quantized-float32.data", float32ParamsContainer, config.paramSize);
  loadParameters<int64_t>(paramsDir + "params-quantized-int64.data", int64ParamsContainer, config.paramSize);
  loadParameters<int8_t>(paramsDir + "params-quantized-int8.data", int8ParamsContainer, config.paramSize);

  /// Fill data into containers
  /// Get user message.
  std::string inputStr;
  getUserInput(inputStr);
  Text<size_t, 2> inputContainer(inputStr);
  tokenizeInput(vocabDir, inputContainer, config.maxTokenLength);
  if constexpr(debug) {
    inputContainer.setTokenCnt(0);
    inputContainer.appendTokenIdx(0);
  }

  /// Run LLaMA Inference
  //  - Perform the forward function.
  //  - Find and append the generated token.
  //  - Continue iterating until the terminal condition is met.
  int generateLen = config.maxTokenLength - inputContainer.getTokenCnt();
  for (int i = 0; i < generateLen; i++) {
    const auto inferenceStart = std::chrono::high_resolution_clock::now();
    // Execute the forward pass of the model.
    _mlir_ciface_forward(resultContainer,
                        &float32ParamsContainer,
                        &int64ParamsContainer,
                        &int8ParamsContainer,
                        &inputContainer);
    
    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;
    // Determine the generated token.
    int tokenIndex = inputContainer.getTokenCnt() - 1;
    const float *startPtr =
        resultContainer[1].getData() + tokenIndex * config.maxVocabSize;
    const float *endPtr = startPtr + config.maxVocabSize;
    int maxIndex;
    if constexpr(debug) {
      maxIndex = i + 1;
    }
    else { 
      maxIndex = findMaxIndex(startPtr, endPtr);
    }
    std::string tok = inputContainer.getStr(maxIndex);
    // Print the generated token and inference time.
    printIterInfo(i, tok, inferenceTime.count() / 1000);

    if constexpr(debug) {
      printLogLabel();
      std::cout << "[DEBUG] output[0]: ";
      for (int i = 0; i < 10; i++) {
        std::cout << resultContainer[0].getData()[i] << " ";
      }
      std::cout << std::endl;
      printLogLabel();
      std::cout << "[DEBUG] output[1]: ";
      for (int i = 0; i < 10; i++) {
        std::cout << startPtr[i] << " ";
      }
      std::cout << std::endl;

      const float *expStartPtr = expectedOutputContainer.getData() + tokenIndex * config.maxVocabSize;
      for (int t = 0; t < config.maxVocabSize; t++) {
        const float error = std::abs(expStartPtr[t] - startPtr[t]);
        if (error > std::abs(0.05 * startPtr[t])) {
          printLogLabel();
          std::cout << "result at iter " << i << ", token " << t << " is " << expStartPtr[t] << ", but expcted " << startPtr[t] << std::endl;
          return 0;
        }
      }
    }

    // Stop if a separator token (2, </s>) or line break token (13 <0x0A>) is
    // generated.
    if (!debug && maxIndex == 2) {
      break;
    }
    // Append the generated token into the input and output container.
    inputContainer.appendTokenIdx(maxIndex);
    outputContainer.appendTokenIdx(maxIndex);
    free(resultContainer[0].release());
    free(resultContainer[1].release());
  }

  /// Print the final result
  // std::cout << "\n\033[33;1m[Input]\033[0m " << inputStr << std::endl;
  std::cout << "\033[33;1m[Output]\033[0m " << outputContainer.revertLlama()
            << std::endl;

  return 0;
}