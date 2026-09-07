//===- FakeWhisperKernel.cpp - Tiny Whisper runtime test kernel ----------===//
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

#include "buddy/Core/Container.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace {

constexpr std::size_t kMaxTokenLength = 4;
constexpr std::size_t kVocabSize = 3;
constexpr float kMutatedAudioMarker = 1234567.0F;

} // namespace

extern "C" void _mlir_ciface_forward(MemRef<float, 3> *outputs,
                                     MemRef<float, 1> *,
                                     MemRef<float, 3> *audio,
                                     MemRef<std::size_t, 2> *tokens) {
  const bool freshAudio = audio->getData()[0] != kMutatedAudioMarker;
  const bool firstStep = tokens->getData()[1] == 0;

  outputs[0] = MemRef<float, 3>(std::vector<std::size_t>{1, 1, 1}, 0.0F);
  outputs[1] = MemRef<float, 3>(
      std::vector<std::size_t>{1, kMaxTokenLength, kVocabSize}, -10.0F);

  const std::size_t step = firstStep ? 0 : 1;
  const std::size_t token = !freshAudio ? 0 : (firstStep ? 1 : 2);
  outputs[1].getData()[step * kVocabSize + token] = 10.0F;

  // A second decode step only succeeds if the runtime reruns preprocessing.
  audio->getData()[0] = kMutatedAudioMarker;
}
