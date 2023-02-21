//===- AudioContainer.h ---------------------------------------------------===//
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
// Audio container descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef FRONTEND_INTERFACES_BUDDY_DAP_AUDIOCONTAINER
#define FRONTEND_INTERFACES_BUDDY_DAP_AUDIOCONTAINER

#include "AudioFile.h"
#include "buddy/Core/Container.h"

namespace dap {

// Audio container.
// - T represents the type of the elements.
// - N represents the number of audio channels (Normally would be 1 or 2).
// If N is smaller than channels from the file, only previous N channels will be
// manipulated.
template <typename T, size_t N> class Audio {
public:
  Audio() : audioFile(), data(nullptr) {}
  explicit Audio(std::string filename) : audioFile(filename), data(nullptr) {}
  void fetchMetadata(const AudioFile<T> &aud);
  bool save(std::string filename);
  AudioFile<T> &getAudioFile() {
    moveToAudioFile();
    return audioFile;
  }
  MemRef<T, N> &getMemRef() {
    moveToMemRef();
    return *data;
  }

protected:
  void moveToMemRef();
  void moveToAudioFile();
  AudioFile<T> audioFile;
  MemRef<T, N> *data;
};

template <typename T, size_t N> bool Audio<T, N>::save(std::string filename) {
  if (!this->audioFile.samples) {
    auto temp = this->data->release();
    if constexpr (std::is_same_v<T, float>) {
      for (int i = 0; i < audioFile.numSamples; i++) {
        if (temp[i] != temp[i]) { // To handle NaN values
          temp[i] = 0.9999999;
        } else { // Clamp the values between -1.0 to 1.0
          temp[i] = std::clamp(temp[i], float(-1.0), float(0.9999999));
        }
      }
    }
    this->audioFile.samples.reset(temp);
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
  if (data)
    delete data;
  intptr_t sizes[N];
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

#endif // FRONTEND_INTERFACES_BUDDY_DAP_AUDIOCONTAINER
