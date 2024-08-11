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
#include <string_view>
#include <cstring>
#include <iomanip>

#include <fp16.h>

using namespace buddy;

#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)

#ifndef LLAMA_DTYPE
#define LLAMA_DTYPE fp16
#endif
constexpr char dtype[] = STR(LLAMA_DTYPE);

constexpr static bool debug = false;

// llama-2-7b
// constexpr size_t ParamsSize = 6771970048;
// constexpr size_t MaxVocabSize = 32000;
// constexpr size_t MaxTokenLength = 40;
// constexpr size_t HiddenSize = 4096;

// tiny-random-llama
// constexpr size_t ParamsSize = 108368;
// constexpr size_t MaxVocabSize = 3000;
// constexpr size_t MaxTokenLength = 40;
// constexpr size_t HiddenSize = 16;

// tiny-llama-1.1b
constexpr size_t ParamsSize = 1105815552;
constexpr size_t MaxVocabSize = 32000;
constexpr size_t MaxTokenLength = 40;
constexpr size_t HiddenSize = 2048;


using fp16_t = uint16_t;
using bf16_t = uint16_t;
using half_t = uint16_t;

/// Declare LLaMA forward function.
extern "C" void _mlir_ciface_forward(MemRef<float, 3> *, MemRef<half_t, 1> *,
                                     Text<size_t, 2> *);

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

// ----------------- FP16/BP16 <-> F32 conversion utils & builtins ----------------------

static float fp162float(fp16_t hf) {
    return fp16_ieee_to_fp32_value(hf);
}

static fp16_t float2fp16(float f) {
  return fp16_ieee_from_fp32_value(f);
}

// Converts the 16 bit representation of a bfloat value to a float value. This
// implementation is adapted from Eigen.
union Float32Bits {
  uint32_t u;
  float f;
};
static float bf162float(bf16_t bfloatBits) {
  Float32Bits floatBits;
  floatBits.u = static_cast<uint32_t>(bfloatBits) << 16;
  return floatBits.f;
}

static bf16_t float2bf16(float f) {
  Float32Bits floatBits;
  floatBits.f = f;
  return static_cast<bf16_t>(floatBits.u >> 16);
}

static half_t float2half(float f) {
  if constexpr(std::string_view(dtype) == "fp16") {
    return float2fp16(f);
  } else if constexpr (std::string_view(dtype) == "bf16") {
    return float2bf16(f);
  }
  assert(false);
}

static float half2float(half_t hf) {
  if constexpr(std::string_view(dtype) == "fp16") {
    return fp162float(hf);
  } else if constexpr (std::string_view(dtype) == "bf16") {
    return bf162float(hf);
  }
  assert(false);
}

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
                    MemRef<half_t, 1> &params) {
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
  float *param_cache = (float*)malloc(sizeof(float) * ParamsSize);
  paramFile.read(reinterpret_cast<char *>(param_cache),
                 sizeof(float) * ParamsSize);
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
  printLogLabel();
  std::cout << "Casting float params to " << STR(LLAMA_DTYPE) << std::endl;
  for (size_t i = 0; i < ParamsSize; i++) {
    params[i] = float2half(param_cache[i]);
  }
  free(param_cache);
  const auto castEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> castTime =
      castEnd - loadEnd;
  printLogLabel();
  std::cout << "Params cast time: " << (double)(loadTime.count()) / 1000
            << "s\n"
            << std::endl;

  // for debug: output loaded param
  half_t *param_data = params.getData();
  printLogLabel();
  std::cout << "[DEBUG] loaded params: " << std::endl;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      std::cout << std::setprecision(10) << half2float(param_data[10 * i + j]) << " ";
    }
    std::cout << std::endl;
  }
}

/// Load float parameters into data container.
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
  // for debug: output loaded param
  float *param_data = params.getData();
  printLogLabel();
  std::cout << "[DEBUG] loaded params: ";
  for (int i = 0; i < 10; i++) {
    std::cout << param_data[i] << " ";
  }
  std::cout << std::endl;
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
  std::cout << "\033[33;1m" << title << "with data type " << dtype << "\033[0m" << std::endl;
  if constexpr(debug) {
    std::cout << "\033[33;1m" << "Debug mode" << "\033[0m" << std::endl;
  }

  /// Define directories of vacabulary and parameter file.
  const std::string vocabDir = "../examples/BuddyF16Llama/vocab.txt";
  const std::string paramsDir = "../examples/BuddyF16Llama/params.data";

  /// Initialize data containers
  Text<size_t, 2> outputContainer;
  MemRef<float, 3> resultContainer[2] = {
      MemRef<float, 3>({1, MaxTokenLength, HiddenSize}, false, 0),
      MemRef<float, 3>({1, MaxTokenLength, MaxVocabSize}, false, 0)};
  MemRef<half_t, 1> paramsContainer({ParamsSize});
  MemRef<float, 1> expectedOutputContainer({MaxTokenLength * MaxVocabSize});
  
  outputContainer.loadVocab(vocabDir);
  loadParameters(paramsDir, paramsContainer);

  if constexpr(debug) {
    const std::string outputDir = "../examples/BuddyF16Llama/" STR(LLAMA_DTYPE) "-output.data";
    loadParameters(outputDir, expectedOutputContainer);
  }

  std::string inputStr;
  getUserInput(inputStr);
  Text<size_t, 2> inputContainer(inputStr);
  tokenizeInput(vocabDir, inputContainer);
  if constexpr(debug) {
    inputContainer.setTokenCnt(0);
    inputContainer.appendTokenIdx(0);
    assert (inputContainer.getTokenCnt() == 1);
  }

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
        resultContainer[1].getData() + tokenIndex * MaxVocabSize;
    const float *endPtr = startPtr + MaxVocabSize;
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
      const float *expStartPtr = expectedOutputContainer.getData() + tokenIndex * MaxVocabSize;

      // printLogLabel();
      // std::cout << "[DEBUG] output[0]: ";
      // for (int i = 0; i < 10; i++) {
      //   std::cout << resultContainer[0].getData()[i] << " ";
      // }
      // std::cout << std::endl;
      printLogLabel();
      std::cout << "[DEBUG] output[1]: ";
      for (int i = 0; i < 10; i++) {
        std::cout << startPtr[i] << " ";
      }
      std::cout << std::endl;
      printLogLabel();
      std::cout << "[DEBUG] expected: ";
      for (int i = 0; i < 10; i++) {
        std::cout << expStartPtr[i] << " ";
      }
      std::cout << std::endl;

      for (int t = 0; t < MaxVocabSize; t++) {
        const float error = std::abs(expStartPtr[t] - startPtr[t]);
        if (error > std::abs(10. * startPtr[t]) && error > 1e-3) {
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
  std::cout << "\n\033[33;1m[Input]\033[0m " << inputStr << std::endl;
  std::cout << "\033[33;1m[Output]\033[0m " << outputContainer.revertLlama()
            << std::endl;

  return 0;
}
