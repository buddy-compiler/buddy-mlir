//===- KokoroRunner.h - Kokoro-82M TTS runner ----------------------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_KOKORORUNNER_H
#define BUDDY_RUNTIME_MODELS_KOKORORUNNER_H

#include "buddy/runtime/core/InferenceRunner.h"

namespace buddy {
namespace runtime {

/// Kokoro-82M text-to-speech runner.
///
/// The compiled kernel produces a speech waveform from phoneme input_ids and a
/// reference style embedding. The runner owns weight loading, construction of
/// the fixed-shape input tensors (deterministic phoneme ids and style buffer),
/// kernel invocation via `_mlir_ciface_forward`, and emission of the generated
/// waveform samples.
class KokoroRunner : public InferenceRunner {
public:
  void run(const RunConfig &cfg) override;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_KOKORORUNNER_H
