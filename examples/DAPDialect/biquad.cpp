//===- biquad.cpp - Example of DAP Biquad Filter --------------------------===//
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
// An end-to-end example of a biquad operation in buddy-mlir.
//
//===----------------------------------------------------------------------===//

#include <buddy/DAP/DAP.h>
#include <chrono>
#include <iostream>

using namespace dap;
using namespace std;

// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

int main() {
  // Print the title of this example.
  const std::string title = "Biquad Operation Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  // Generate the kernel for a biquad filter operation.
  // Params:
  //   Input kernel: Stores generated kernel data.
  //   Frequency: Normalized frequency (frequency_Hz / samplerate_Hz).
  //   Quality factor: Defines the filter's bandwidth relative to its
  //        center frequency.
  intptr_t kernelSize = 6;
  MemRef<float, 1> kernel(&kernelSize);
  dap::biquadLowpass<float, 1>(kernel, /*frequency=*/0.3, /*Q=*/-1.0);

  // Initialize data containers.
  // Params:
  //    Input container: Stores the raw audio data.
  // Returns:
  //    Output memory reference: Provides a MemRef for saving the output.
  Audio<float, 1> inputContainer("../../tests/Interface/core/TestAudio.wav");
  intptr_t samplesNum = static_cast<intptr_t>(inputContainer.getSamplesNum());
  MemRef<float, 1> outputMemRef(&samplesNum);

  // Apply the biquad filter operation to the audio data.
  printLogLabel();
  std::cout << "Running biquad operation..." << std::endl;
  const auto loadStart = std::chrono::high_resolution_clock::now();
  dap::biquad(&inputContainer, &kernel, &outputMemRef);
  const auto loadEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> loadTime =
      loadEnd - loadStart;
  printLogLabel();
  std::cout << "Audio processing time: " << (double)(loadTime.count()) / 1000
            << "s\n"
            << std::endl;

  // Convert a MemRef object to an Audio object and set the metadata.
  Audio<float, 1> outputContainer(std::move(outputMemRef));
  outputContainer.setBitDepth(inputContainer.getBitDepth());
  outputContainer.setSamplesNum(inputContainer.getSamplesNum());
  outputContainer.setChannelsNum(inputContainer.getChannelsNum());
  outputContainer.setSampleRate(inputContainer.getSampleRate());

  // Save the processed data to an audio file.
  std::string saveFileName = "BiquadTestAudio.wav";
  outputContainer.saveToFile(saveFileName, "wave");
  printLogLabel();
  std::cout << "Processed audio data saved in: " << saveFileName << "\n"
            << std::endl;

  return 0;
}
