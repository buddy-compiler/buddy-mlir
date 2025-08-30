//===- whisper-main.cpp ---------------------------------------------------===//
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
//
// This file implements an example for Whisper Model Inference. 
//
// ------------------------------------------------------------------------===//

#include <buddy/Core/Container.h>
#include <buddy/DAP/DAP.h>
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace std;
using namespace buddy;
using namespace dap;

constexpr size_t ParamsSize = 72593920;
constexpr size_t MaxVocabSize = 51865;
constexpr size_t MaxTokenLength = 448;

/// Declare Whisper forward function.
extern "C" void _mlir_ciface_forward(MemRef<float, 3> *, MemRef<float, 1> *,
                                     MemRef<float, 3> *, MemRef<size_t, 2> *);

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

/// Print information for each iteration.
void printIterInfo(size_t iterIdx, std::string str, double time) {
  std::cout << "\033[32;1m[Iteration " << iterIdx << "] \033[0m";
  std::cout << "Token: " << str << " | "
            << "Time: " << time << "s" << std::endl;
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

/// Conduct audio data preprocess.
void runPreprocess(dap::Audio<double, 1> &rawAudioContainer,
                   MemRef<float, 3> &audioFeatures) {
  printLogLabel();
  std::cout << "Preprocessing audio..." << std::endl;
  const auto loadStart = std::chrono::high_resolution_clock::now();
  dap::whisperPreprocess(&rawAudioContainer, &audioFeatures);
  const auto loadEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> loadTime =
      loadEnd - loadStart;
  printLogLabel();
  std::cout << "Audio preprocess time: " << (double)(loadTime.count()) / 1000
            << "s\n"
            << std::endl;
}

/// Find the index of the max value.
int findMaxIndex(const float *start, const float *end) {
  return std::distance(start, std::max_element(start, end));
}

// -----------------------------------------------------------------------------
// Whisper Inference Main Entry
// -----------------------------------------------------------------------------

int main() {

  /// Print the title of this example.
  const std::string title = "Whisper Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  /// Define directories of vacabulary and parameter file.
  std::string whisperDir = WHISPER_EXAMPLE_PATH;
  std::string whisperBuildDir = WHISPER_EXAMPLE_BUILD_PATH;
  const std::string vocabDir =  whisperDir + "/vocab.txt";
  const std::string paramsDir =  whisperBuildDir + "/arg0.data";

  /// Initialize data containers
  //  - Result container
  //  - Output container.
  //  - Parameters container.
  Text<size_t, 2> outputContainer;
  Audio<double, 1> rawAudioContainer(whisperDir + "/audio.wav");
  MemRef<float, 3> audioInput({1, 80, 3000});
  MemRef<float, 3> resultContainer[2] = {
      MemRef<float, 3>({1, 1500, 512}, false, 0),
      MemRef<float, 3>({1, 448, MaxVocabSize}, false, 0),
  };
  MemRef<size_t, 2> textContainer({1, MaxTokenLength}, 50258);
  MemRef<float, 1> paramsContainer({ParamsSize});

  /// Fill data into containers
  //  - Output: register vocabulary.
  //  - Parameters: load parameters from the `arg0` file into the container.
  //  - Input: compute audioInput.
  outputContainer.loadVocab(vocabDir);
  loadParameters(paramsDir, paramsContainer);
  runPreprocess(rawAudioContainer, audioInput);

  /// Run Whisper Inference
  //  - Perform the forward function.
  //  - Find and append the generated token.
  //  - Continue iterating until the terminal condition is met.

  for (size_t i = 0; i < MaxTokenLength - 1; i++) {
    const auto inferenceStart = std::chrono::high_resolution_clock::now();
    // Execute the forward pass of the model.
    _mlir_ciface_forward(resultContainer, &paramsContainer, &audioInput,
                         &textContainer);
    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;

    // Determine the generated token.
    const float *startPtr = resultContainer[1].getData() + i * MaxVocabSize;
    const float *endPtr = startPtr + MaxVocabSize;

    int maxIndex = findMaxIndex(startPtr, endPtr);
    std::string tok = outputContainer.getStr(maxIndex);
    // Print the generated token and inference time.
    printIterInfo(i, tok, inferenceTime.count() / 1000);

    // Stop if the end token (50257, <|endoftext|>) is generated.
    if (maxIndex == 50257) {
      break;
    }
    // Append the generated token into the output container.
    textContainer.getData()[i + 1] = maxIndex;
    outputContainer.appendTokenIdx(maxIndex);

    free(resultContainer[0].release());
    free(resultContainer[1].release());
  }

  /// Print the final result
  std::cout << "\033[33;1m[Output]\033[0m " << outputContainer.revertWhisper()
            << std::endl;

  return 0;
}