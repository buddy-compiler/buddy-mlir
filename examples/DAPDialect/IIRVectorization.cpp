//===- IIRVectorization.cpp - Example of DAP IIR Vectorization ------------===//
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
// This file implements an end to end example for iir filter in buddy-mlir. It
// generates coefficients for a filter and apply it on a piece of mono audio,
// then saves the audio.
// This file will be linked with the object file which use dap vectorization
// pass to generate the executable file.
//
//===----------------------------------------------------------------------===//

#include <buddy/DAP/DAP.h>
#include <iostream>

using namespace dap;
using namespace std;

int main(int argc, char *argv[]) {
  string fileName = "../../tests/Interface/core/NASA_Mars.wav";
  string saveFileName = "IIR_VECTORIZATION_PASS_NASA_Mars.wav";
  if (argc >= 2) {
    fileName = argv[1];
  }
  if (argc == 3) {
    saveFileName = argv[2];
  }
  cout << "Usage: IIRVectorizationPass [loadPath] [savePath]" << endl;
  cout << "Current specified path: \n";
  cout << "Load: " << fileName << endl;
  cout << "Save: " << saveFileName << endl;
  // Order for butterworth filter.
  int order = 8;
  // Each SOS matrix has 6 paramters.
  intptr_t kernelSize[2] = {int(order / 2), 6};
  MemRef<float, 2> kernel(kernelSize);
  // cutoff frequency = 1000, fs = 48000.
  dap::iirLowpass<float, 2>(kernel, dap::butterworth<float>(order), 1000,
                            48000);

  auto aud = dap::Audio<float, 1>(fileName);
  aud.getAudioFile().printSummary();
  dap::Audio<float, 1> output;
  output.fetchMetadata(aud.getAudioFile());
  output.getAudioFile().setAudioBuffer(nullptr);

  dap::IIR(&aud.getMemRef(), &kernel, &output.getMemRef(),
           /*isVectorization=*/true);

  cout << "Saving file:" << endl;
  cout << (output.save(saveFileName) ? "OK" : "ERROR") << endl;

  return 0;
}
