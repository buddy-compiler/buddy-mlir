//===- WhisperRuntime.h - Reusable long-lived Whisper runtime ------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_WHISPERRUNTIME_H
#define BUDDY_RUNTIME_MODELS_WHISPERRUNTIME_H

#include "buddy/runtime/core/AudioTranscriptionTypes.h"

#include <functional>
#include <memory>
#include <string>

namespace buddy {
namespace runtime {

struct WhisperProgress {
  std::size_t iteration = 0;
  int tokenId = -1;
  std::string token;
  double inferenceSeconds = 0.0;
};

using WhisperProgressCallback = std::function<void(const WhisperProgress &)>;

/// Owns the compiled model library, vocabulary and resident weights. load()
/// and transcribe() are internally serialized; all request-local decode and
/// audio state is discarded before the next request starts.
class WhisperRuntime {
public:
  WhisperRuntime();
  ~WhisperRuntime();
  WhisperRuntime(const WhisperRuntime &) = delete;
  WhisperRuntime &operator=(const WhisperRuntime &) = delete;

  void load(const AudioTranscriptionModelConfig &config);
  AudioTranscriptionResult
  transcribe(const AudioInput &audio, int maxTokens,
             const WhisperProgressCallback &progress = nullptr);

  bool isLoaded() const;
  std::string modelName() const;
  std::size_t maxTokenLength() const;
  std::string modelSoPath() const;
  std::string weightsPath() const;
  std::string vocabPath() const;
  double weightLoadSeconds() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_WHISPERRUNTIME_H
