//===- AudioContainerTest.cpp ---------------------------------------------===//
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
// This is the audio container test file.
//
//===----------------------------------------------------------------------===//

// RUN: buddy-audio-container-test 2>&1 | FileCheck %s

#include <buddy/DAP/AudioContainer.h>
#include <iostream>

using namespace std;

int main() {
  dap::Audio<float, 1> aud("../../../../tests/Interface/core/NASA_Mars.wav");
  auto &audioFile = aud.getAudioFile();
  // CHECK: 1
  fprintf(stderr, "%u\n", audioFile.getNumChannels());
  // CHECK: 24
  fprintf(stderr, "%u\n", audioFile.getBitDepth());
  // CHECK: 2000000
  fprintf(stderr, "%u\n", audioFile.getNumSamplesPerChannel());
  // CHECK: 100000
  fprintf(stderr, "%u\n", audioFile.getSampleRate());

  return 0;
}
