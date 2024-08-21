//===- IIRLowpass.cpp - Example of DAP IIR Filter -------------------------===//
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
// An end-to-end example of the scalar version IIR (Infinite Impulse Response)
// operation in buddy-mlir.
//
//===----------------------------------------------------------------------===//

#include <buddy/DAP/DAP.h>
#include <chrono>
#include <iostream>

using namespace dap;
using namespace std;

// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

int main(int argc, char *argv[]) {
  // Print the title of this example.
  const std::string title =
      "Scalar Version IIR Operation Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  // Allocate kernel MemRef for an IIR filter operation.
  // Params:
  //   Order: The order of the butterworth filter.
  //   Parameter size: Each SOS matrix has 6 parameters.
  int order = 8;
  intptr_t kernelSize[2] = {int(order / 2), 6};
  MemRef<float, 2> kernel(kernelSize);

  // Generate the kernel for an IIR filter operation.
  // Params:
  //   Input kernel: Stores generated kernel data.
  //   Lowpass filter: Supports butterworth filter upto order 12 for now.
  //   Lowpass frequency: The lowpass cutoff frequency.
  //   Sampling frequency: The rate at which the input data is sampled.
  dap::iirLowpass<float, 2>(/*kernel=*/kernel,
                            /*filter=*/dap::butterworth<float>(order),
                            /*frequency=*/1000,
                            /*fs=*/48000);

  // Initialize data containers.
  // Params:
  //    Input container: Stores the raw audio data.
  // Returns:
  //    Output memory reference: Provides a MemRef for saving the output.
  Audio<float, 1> inputContainer("../../tests/Interface/core/TestAudio.wav");
  intptr_t samplesNum = static_cast<intptr_t>(inputContainer.getSamplesNum());
  MemRef<float, 1> outputMemRef(&samplesNum);

  // Apply scalar version IIR operation to the audio data.
  printLogLabel();
  std::cout << "Running scalar version IIR operation..." << std::endl;
  const auto loadStart = std::chrono::high_resolution_clock::now();
  dap::IIR(&inputContainer, &kernel, &outputMemRef);
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
  std::string saveFileName = "ScalarVersionIIRTestAudio.wav";
  outputContainer.saveToFile(saveFileName, "wave");
  printLogLabel();
  std::cout << "Processed audio data saved in: " << saveFileName << "\n"
            << std::endl;

  return 0;
}
