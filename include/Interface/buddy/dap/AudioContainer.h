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

#include "AudioFile.h"
#include "Interface/buddy/core/Container.h"

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

} // namespace dap
#include "Interface/core/AudioContainer.cpp"

#endif // INTERFACE_BUDDY_CORE_AUDIOCONTAINER
