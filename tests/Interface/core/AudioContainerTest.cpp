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
  dap::Audio<float, 1> aud("../../../../tests/Interface/core/TestAudio.wav");
  // CHECK: WAV
  fprintf(stderr, "%s\n", aud.getFormatName().c_str());
  // CHECK: 16
  fprintf(stderr, "%d\n", aud.getBitDepth());
  // CHECK: 77040
  fprintf(stderr, "%lu\n", aud.getSamplesNum());
  // CHECK: 1
  fprintf(stderr, "%d\n", aud.getChannelsNum());
  // CHECK: 16000
  fprintf(stderr, "%d\n", aud.getSampleRate());
  // CHECK: -0.000153
  fprintf(stderr, "%f\n", aud.getData()[3]);

  return 0;
}
