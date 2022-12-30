//===- AudioContainer.cpp -------------------------------------------------===//
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
// This file implements the audio container descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef CORE_AUDIO_CONTAINER_DEF
#define CORE_AUDIO_CONTAINER_DEF

#include "Interface/buddy/dap/AudioContainer.h"
#include "Interface/buddy/core/Container.h"

namespace dap {

template <typename T, size_t N> bool Audio<T, N>::save(std::string filename) {
  if (!this->audioFile.samples) {
    this->audioFile.samples.reset(this->data->release());
  }
  return this->audioFile.save(filename);
}

template <typename T, size_t N>
void Audio<T, N>::fetchMetadata(const AudioFile<T> &aud) {
  this->audioFile.setBitDepth(aud.getBitDepth());
  this->audioFile.setSampleRate(aud.getSampleRate());
  this->audioFile.numSamples = aud.numSamples;
  this->audioFile.numChannels = aud.numChannels;
  this->audioFile.setAudioBuffer(nullptr);
}
template <typename T, size_t N> void Audio<T, N>::moveToMemRef() {
  if(data) delete data;
  size_t sizes[N];
  for (size_t i = 0; i < N; ++i) {
    sizes[i] = audioFile.numSamples;
  }
  data = new MemRef<T, N>(audioFile.samples, sizes);
}
template <typename T, size_t N> void Audio<T, N>::moveToAudioFile() {
  if (data) {
    auto temp = data->release();
    audioFile.setAudioBuffer(temp);
  }
}
} // namespace dap

#endif // CORE_AUDIO_CONTAINER_DEF
