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
#include <cstdlib>
#include <vector>

namespace {

constexpr std::size_t kMaxTokenLength = 4;
constexpr std::size_t kVocabSize = 3;
constexpr float kMutatedMarker = 1234567.0F;

std::size_t encoderCalls = 0;
std::size_t decoderCalls = 0;

// The runtime must hand the compiled functions empty result descriptors;
// assigning over an allocated one would leak it.
void requireEmptyResult(MemRef<float, 3> *result) {
  if (result->getData())
    std::abort();
}

} // namespace

extern "C" std::size_t whisper_fake_encoder_calls() { return encoderCalls; }
extern "C" std::size_t whisper_fake_decoder_calls() { return decoderCalls; }

extern "C" void _mlir_ciface_forward_encoder(MemRef<float, 3> *states,
                                             MemRef<float, 1> *,
                                             MemRef<float, 3> *audio) {
  ++encoderCalls;
  requireEmptyResult(states);
  *states = MemRef<float, 3>(std::vector<std::size_t>{1, 1, 1}, 0.0F);
  audio->getData()[0] = kMutatedMarker;
}

extern "C" void _mlir_ciface_forward_decoder(MemRef<float, 3> *logits,
                                             MemRef<float, 1> *,
                                             MemRef<std::size_t, 2> *tokens,
                                             MemRef<float, 3> *states) {
  ++decoderCalls;
  requireEmptyResult(logits);
  const bool freshStates = states->getData()[0] != kMutatedMarker;
  const bool firstStep = tokens->getData()[1] == 0;

  *logits = MemRef<float, 3>(
      std::vector<std::size_t>{1, kMaxTokenLength, kVocabSize}, -10.0F);

  const std::size_t step = firstStep ? 0 : 1;
  const std::size_t token = !freshStates ? 0 : (firstStep ? 1 : 2);
  logits->getData()[step * kVocabSize + token] = 10.0F;

  // A second decode step only succeeds if the runtime gives each step its own
  // copy of the encoder states.
  states->getData()[0] = kMutatedMarker;
}
