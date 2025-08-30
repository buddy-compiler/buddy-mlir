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

#include "AudioFile.h"
#include <buddy/DAP/AudioContainer.h>
#include <iostream>

using namespace std;

int main() {
  // ---------------------------------------------------------------------------
  // 1. Print Decoded Reuslts using Buddy Audio Container
  // ---------------------------------------------------------------------------

  // Read and decode audio file with Buddy Audio Container.
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
  // CHECK: -0.000275
  fprintf(stderr, "%f\n", aud.getData()[4]);

  // ---------------------------------------------------------------------------
  // 2. Compare Encoded results using Buddy Audio Container and AudioFile.h
  // ---------------------------------------------------------------------------

  // Encode the audio data and save it to a file using the Buddy Audio Container
  string filePath = "./buddyEncodeResult.wav";
  aud.saveToFile(filePath, "WAVE");

  // Print metadata and sample values using the Buddy Audio Container.
  dap::Audio<float, 1> audContainer(filePath);
  // CHECK: 16
  fprintf(stderr, "%d\n", audContainer.getBitDepth());
  // CHECK: 77040
  fprintf(stderr, "%lu\n", audContainer.getSamplesNum());
  // CHECK: 1
  fprintf(stderr, "%d\n", audContainer.getChannelsNum());
  // CHECK: 16000
  fprintf(stderr, "%d\n", audContainer.getSampleRate());
  // CHECK: -0.000122
  fprintf(stderr, "%f\n", audContainer.getData()[3]);
  // CHECK: -0.000244
  fprintf(stderr, "%f\n", audContainer.getData()[4]);

  // Print metadata and sample values using the third-party (AudioFile.h).
  AudioFile<float> audFile(filePath);
  // CHECK: 16
  fprintf(stderr, "%d\n", audFile.getBitDepth());
  // CHECK: 77040
  fprintf(stderr, "%d\n", audFile.getNumSamplesPerChannel());
  // CHECK: 1
  fprintf(stderr, "%d\n", audFile.getNumChannels());
  // CHECK: 16000
  fprintf(stderr, "%d\n", audFile.getSampleRate());
  // CHECK: -0.000122
  fprintf(stderr, "%f\n", audFile.getSample(0, 3));
  // CHECK: -0.000244
  fprintf(stderr, "%f\n", audFile.getSample(0, 4));

  return 0;
}
