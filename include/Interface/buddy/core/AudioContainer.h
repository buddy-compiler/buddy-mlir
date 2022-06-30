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

#ifndef INTERFACE_BUDDY_CORE_AUDIOCONTAINER
#define INTERFACE_BUDDY_CORE_AUDIOCONTAINER

#include "Interface/buddy/core/Container.h"
#include "Interface/buddy/dap/AudioFile.h"

// Audio container.
// - T represents the type of the elements.
// - N represents the number of audio channels (Normally would be 1 or 2).
template <typename T, size_t N> class Audio : public MemRef<T, N> {
public:
  using AudioBuffer = std::vector<std::vector<T>>;
  Audio(std::string path) {
    // TODO: check if file is correctly loaded.
    audioFile = new AudioFile<T>(path);
    // TODO: support multi-channel audio
    if (audioFile->getNumChannels() != N)
      assert(0 && "This audio container doesn't match the channels for the "
                  "file specified.");
    loadFromAudioBuffer(audioFile->samples);
  }
  void loadFromAudioBuffer(AudioBuffer audioBuffer) {
    for (size_t i = 0; i < audioBuffer.size(); i++) {
      this->sizes[i] = audioBuffer[i].size();
    }

    this->size = this->product(this->sizes);
    this->allocated = new T[this->size];
    this->aligned = this->allocated;
    for (size_t i = 0; i < audioBuffer.size(); ++i) {
      for (size_t j = 0; j < audioBuffer[i].size(); ++j) {
        this->aligned[i * this->sizes[i] + j] = audioBuffer[i][j];
      }
    }
    this->setStrides();
  }
  AudioFile<T> &getAudioFile() { return *audioFile; }
  void setAudioFile(AudioFile<T> *aud) { audioFile = aud; }
  explicit Audio(intptr_t sizes[N], T init = T(0))
      : MemRef<T, N>(sizes, init) {}

  void save(std::string path) {
    if (audioFile->samples.empty())
      assert(0 && "Cannot save empty audio.");
    for (size_t i = 0; i < this->getRank(); ++i) {
      auto &row = audioFile->samples[i];
      for (auto j = 0; j < this->sizes[i]; ++j) {
        row[j] = this->aligned[i * this->sizes[i] + j];
      }
    }
    audioFile->save(path);
    delete audioFile;
  }

private:
  AudioFile<T> *audioFile;
};

#include "Interface/core/AudioContainer.cpp"

#endif // INTERFACE_BUDDY_CORE_AUDIOCONTAINER
