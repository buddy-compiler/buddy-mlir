//===- WhisperPreprocessor.cpp --------------------------------------------===//
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
// An example of the Whisper Preprocessor operation.
//
//===----------------------------------------------------------------------===//

#include <buddy/DAP/DAP.h>
#include <chrono>
#include <fstream>
#include <iostream>

using namespace dap;
using namespace std;

// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

// Write preprocessing results to a text file.
void printResult(MemRef<float, 3> &outputMemRef) {
  ofstream fout("whisperPreprocessResult.txt");
  // Print title.
  fout << "-----------------------------------------" << std::endl;
  fout << "[ Whisper Preprocess Result ]" << std::endl;
  fout << "-----------------------------------------" << std::endl;
  // Print reuslt data.
  for (int i = 0; i < 240000; ++i) {
    fout << outputMemRef[i] << std::endl;
  }
  fout.close();
}

int main() {
  // Print the title of this example.
  const std::string title = "Whisper Preprocess Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  // Initialize data containers.
  // Params:
  //    Input container: Stores raw audio data.
  // Returns:
  //    Output memory reference: Features formatted as memref<1x80x3000xf32>.
  Audio<double, 1> inputContainer("../../examples/BuddyWhisper/audio.wav");
  float *outputAlign = new float[240000];
  intptr_t outputSizes[3] = {1, 80, 3000};
  MemRef<float, 3> outputMemRef(outputAlign, outputSizes);

  // Compute audio features from raw audio data.
  printLogLabel();
  std::cout << "Preprocessing audio..." << std::endl;
  const auto loadStart = std::chrono::high_resolution_clock::now();
  dap::whisperPreprocess(&inputContainer, &outputMemRef);
  const auto loadEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> loadTime =
      loadEnd - loadStart;
  printLogLabel();
  std::cout << "Audio preprocess time: " << (double)(loadTime.count()) / 1000
            << "s\n"
            << std::endl;

  // printResult(outputMemRef);

  return 0;
}
