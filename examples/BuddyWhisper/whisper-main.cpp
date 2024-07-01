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

#include "whisper-main.h"

// -----------------------------------------------------------------------------
// Whisper Inference Main Entry
// -----------------------------------------------------------------------------

int main() {

  /// Print the title of this example.
  const std::string title = "Whisper Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  /// Define directories of vacabulary and parameter file.
  const std::string vocabDir = "../../examples/BuddyWhisper/vocab.txt";
  const std::string paramsDir = "../../examples/BuddyWhisper/arg0.data";

  /// Initialize data containers
  //  - Result container
  //  - Output container.
  //  - Parameters container.
  Text<size_t, 2> outputContainer;
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
  //  - Input: generate audioInput from rawAudioData.
  outputContainer.loadVocab(vocabDir);
  loadParameters(paramsDir, paramsContainer);
  rawAudioData = std::move(MemRef<double, 1>(rawSpeech, inputShape));
  dap::WhisperPreprocess(&rawAudioData, &audioInput);

  /// Run Whisper Inference
  //  - Perform the forward function.
  //  - Find and append the generated token.
  //  - Continue iterating until the terminal condition is met.

  for (int i = 0; i < MaxTokenLength - 1; i++) {
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
