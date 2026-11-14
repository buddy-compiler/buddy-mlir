//===- WhisperAudioTranscriptionModel.h - Whisper serving adapter --------===//
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

#ifndef BUDDY_RUNTIME_MODELS_WHISPERAUDIOTRANSCRIPTIONMODEL_H
#define BUDDY_RUNTIME_MODELS_WHISPERAUDIOTRANSCRIPTIONMODEL_H

#include "buddy/runtime/core/AudioTranscriptionModel.h"

#include <memory>

namespace buddy {
namespace runtime {

class WhisperAudioTranscriptionModel final : public AudioTranscriptionModel {
public:
  WhisperAudioTranscriptionModel();
  ~WhisperAudioTranscriptionModel() override;

  void load(const AudioTranscriptionModelConfig &config) override;
  ModelStatus status() const override;
  AudioTranscriptionResult
  transcribe(const AudioTranscriptionRequest &request) override;

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_WHISPERAUDIOTRANSCRIPTIONMODEL_H
