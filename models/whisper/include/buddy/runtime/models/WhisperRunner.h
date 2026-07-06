//===- WhisperRunner.h - Whisper inference runner -------------------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_WHISPERRUNNER_H
#define BUDDY_RUNTIME_MODELS_WHISPERRUNNER_H

#include "buddy/runtime/core/InferenceRunner.h"

namespace buddy {
namespace runtime {

/// Full inference runner for the Whisper encoder-decoder ASR model.
///
/// Implements the complete loop: load weights → preprocess audio → greedy
/// autoregressive decode over a single `_mlir_ciface_forward` entrypoint →
/// detokenize.  Handles both Mode A (.rax manifest) and Mode B (explicit
/// --model-so / --weights paths).  The audio file comes from cfg.audioPath
/// (--audio); when empty it falls back to `audio.wav` next to the model.
class WhisperRunner : public InferenceRunner {
public:
  void run(const RunConfig &cfg) override;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_WHISPERRUNNER_H
