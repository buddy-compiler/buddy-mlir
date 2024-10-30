//===- RFFT.cpp - Example of DAP RFFT Operation ---------------------------===//
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
// An example of the RFFT function from Whisper Preprocessor operation.
//
//===----------------------------------------------------------------------===//

#include <buddy/DAP/DAP.h>
#include <chrono>
#include <fstream>
#include <iostream>

#define testLength 840

using namespace dap;
using namespace std;

// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

// Write preprocessing results to a text file.
void printResult(MemRef<double, 1> &outputMemRef) {
  ofstream fout("whisperPreprocessResultRFFT.txt");
  // Print title.
  fout << "-----------------------------------------" << std::endl;
  fout << "[ Buddy RFFT Result ]" << std::endl;
  fout << "-----------------------------------------" << std::endl;
  // Print reuslt data.
  for (int i = 0; i < testLength; ++i) {
    fout << outputMemRef[i] << std::endl;
  }
  fout.close();
}

int main() {
  // Print the title of this example.
  const std::string title = "RFFT Operation Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  double *inputAlign = new double[testLength];
  for (int i = 0; i < testLength; ++i) {
    inputAlign[i] = static_cast<double>(i);
  }
  intptr_t inputSizes[1] = {testLength};
  MemRef<double, 1> inputMemRef(inputAlign, inputSizes);

  printLogLabel();
  std::cout << "Running RFFT operation" << std::endl;
  const auto loadStart = std::chrono::high_resolution_clock::now();
  dap::RFFT(&inputMemRef);
  const auto loadEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> loadTime =
      loadEnd - loadStart;
  printLogLabel();
  std::cout << "RFFT time: " << (double)(loadTime.count()) / 1000
            << "s\n"
            << std::endl;

  printResult(inputMemRef);

  return 0;
}
