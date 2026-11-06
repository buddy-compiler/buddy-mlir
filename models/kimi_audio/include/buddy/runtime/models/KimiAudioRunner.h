//===- KimiAudioRunner.h - Kimi-Audio single-forward runner ---------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_KIMIAUDIORUNNER_H
#define BUDDY_RUNTIME_MODELS_KIMIAUDIORUNNER_H

#include "buddy/runtime/core/InferenceRunner.h"

namespace buddy {
namespace runtime {

/// Kimi-Audio-7B-Instruct single-forward runner.
///
/// The compiled kernel performs one fixed-shape forward pass over a padded
/// text sequence (whisper features disabled), producing audio_logits and
/// text_logits. The runner owns Qwen-style BPE tokenization, weight loading,
/// kernel invocation, and emission of the per-position argmax token ids.
class KimiAudioRunner : public InferenceRunner {
public:
  void run(const RunConfig &cfg) override;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_KIMIAUDIORUNNER_H
