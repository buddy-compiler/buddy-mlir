//===- WhisperAudioTranscriptionModel.cpp - Whisper serving adapter ------===//
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

#include "buddy/runtime/models/WhisperAudioTranscriptionModel.h"

#include "buddy/runtime/models/WhisperRuntime.h"

#include <memory>
#include <mutex>
#include <stdexcept>

namespace buddy {
namespace runtime {

class WhisperAudioTranscriptionModel::Impl {
public:
  void load(const AudioTranscriptionModelConfig &config) {
    {
      std::lock_guard<std::mutex> lock(statusMutex);
      statusValue = {};
      statusValue.state = ModelLoadState::Loading;
      statusValue.modelName =
          config.modelName.empty() ? "whisper_base" : config.modelName;
      statusValue.backend = "cpu";
      statusValue.contextLength = 448;
      statusValue.message = "model is loading";
    }

    try {
      runtime.load(config);
      std::lock_guard<std::mutex> lock(statusMutex);
      statusValue.state = ModelLoadState::Ready;
      statusValue.modelName = runtime.modelName();
      statusValue.contextLength = runtime.maxTokenLength();
      statusValue.message = "model loaded";
    } catch (const std::exception &error) {
      std::lock_guard<std::mutex> lock(statusMutex);
      statusValue.state = ModelLoadState::Error;
      statusValue.message = error.what();
      throw;
    }
  }

  ModelStatus status() const {
    std::lock_guard<std::mutex> lock(statusMutex);
    return statusValue;
  }

  AudioTranscriptionResult
  transcribe(const AudioTranscriptionRequest &request) {
    const ModelStatus current = status();
    if (current.state != ModelLoadState::Ready)
      throw std::runtime_error("Whisper transcription model is not ready");
    if (!request.model.empty() && request.model != current.modelName)
      throw std::invalid_argument("model name does not match loaded model");
    if (request.audio.uri.empty() == request.audio.bytes.empty())
      throw std::invalid_argument(
          "provide exactly one audio source as a local WAV path");
    return runtime.transcribe(request.audio, request.maxTokens);
  }

private:
  mutable std::mutex statusMutex;
  ModelStatus statusValue;
  WhisperRuntime runtime;
};

WhisperAudioTranscriptionModel::WhisperAudioTranscriptionModel()
    : impl(std::make_unique<Impl>()) {}
WhisperAudioTranscriptionModel::~WhisperAudioTranscriptionModel() = default;

void WhisperAudioTranscriptionModel::load(
    const AudioTranscriptionModelConfig &config) {
  impl->load(config);
}

ModelStatus WhisperAudioTranscriptionModel::status() const {
  return impl->status();
}

AudioTranscriptionResult WhisperAudioTranscriptionModel::transcribe(
    const AudioTranscriptionRequest &request) {
  return impl->transcribe(request);
}

} // namespace runtime
} // namespace buddy
