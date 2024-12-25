//===- FIRVectorization.cpp - Example of DAP FIR Vectorization ------------===//
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
// An end-to-end example of the vectorized FIR (Finite Impulse Response)
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

int main() {
  // Print the title of this example.
  const std::string title =
      "Vectorized FIR Operation Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  // Generate the kernel for a FIR filter operation.
  // Params:
  //   Input kernel: Stores generated kernel data.
  //   Type: Specifies the window type from the WINDOW_TYPE enum class.
  //   Length: The length of the filter.
  //   Cutoff: The lowpass cutoff frequency.
  //   Argument: Filter-specific arguments, with size limited by the
  //        WINDOW_TYPE.
  intptr_t kernelSize = 100;
  MemRef<float, 1> kernel(&kernelSize);
  dap::firLowpass<float, 1>(/*input=*/kernel,
                            /*type=*/dap::WINDOW_TYPE::BLACKMANHARRIS7,
                            /*len=*/kernelSize, /*cutoff=*/0.3,
                            /*args=*/nullptr);

  // Initialize data containers.
  // Params:
  //    Input container: Stores the raw audio data.
  // Returns:
  //    Output memory reference: Provides a MemRef for saving the output.
  Audio<float, 1> inputContainer("../../tests/Interface/core/TestAudio.wav");
  intptr_t samplesNum = static_cast<intptr_t>(inputContainer.getSamplesNum());
  MemRef<float, 1> outputMemRef(&samplesNum);

  // Apply vectorized FIR operation to the audio data.
  printLogLabel();
  std::cout << "Running vectorized FIR operation..." << std::endl;
  const auto loadStart = std::chrono::high_resolution_clock::now();
  dap::FIR(&inputContainer, &kernel, &outputMemRef,
           /*isVectorization=*/true);
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
  std::string saveFileName = "VectorizedFIRTestAudio.wav";
  outputContainer.saveToFile(saveFileName, "wave");
  printLogLabel();
  std::cout << "Processed audio data saved in: " << saveFileName << "\n"
            << std::endl;

  return 0;
}
