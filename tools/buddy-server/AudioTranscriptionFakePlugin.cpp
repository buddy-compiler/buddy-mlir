//===- AudioTranscriptionFakePlugin.cpp - Test transcription plugin ------===//
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

#include "buddy/runtime/core/AudioTranscriptionModelPlugin.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>

namespace {

class FakeModel final : public buddy::runtime::AudioTranscriptionModel {
public:
  ~FakeModel() override {
    if (const char *marker =
            std::getenv("BUDDY_TRANSCRIPTION_DESTROY_MARKER")) {
      if (FILE *file = std::fopen(marker, "w")) {
        std::fputs("destroyed", file);
        std::fclose(file);
      }
    }
  }

  void
  load(const buddy::runtime::AudioTranscriptionModelConfig &config) override {
    statusValue.state = buddy::runtime::ModelLoadState::Loading;
    statusValue.modelName =
        config.modelName.empty() ? "fake_transcription" : config.modelName;
    statusValue.backend = "cpu";
    statusValue.contextLength = 448;
    if (const char *delay =
            std::getenv("BUDDY_FAKE_TRANSCRIPTION_LOAD_DELAY_MS"))
      std::this_thread::sleep_for(std::chrono::milliseconds(std::stoi(delay)));
    if (std::getenv("BUDDY_FAKE_TRANSCRIPTION_LOAD_ERROR")) {
      statusValue.state = buddy::runtime::ModelLoadState::Error;
      statusValue.message = "fake load failure";
      throw std::runtime_error(statusValue.message);
    }
    statusValue.state = buddy::runtime::ModelLoadState::Ready;
    statusValue.message = "model loaded";
  }

  buddy::runtime::ModelStatus status() const override { return statusValue; }

  buddy::runtime::AudioTranscriptionResult transcribe(
      const buddy::runtime::AudioTranscriptionRequest &request) override {
    buddy::runtime::AudioTranscriptionResult result;
    result.model = statusValue.modelName;
    result.text = "fake transcription";
    result.generatedTokens =
        static_cast<std::size_t>(request.maxTokens > 1 ? 2 : 1);
    result.timings.preprocessMs = 1.0;
    result.timings.inferenceMs = 2.0;
    result.timings.totalMs = 3.0;
    return result;
  }

private:
  buddy::runtime::ModelStatus statusValue;
};

} // namespace

extern "C" buddy::runtime::AudioTranscriptionModel *
buddy_create_audio_transcription_model_v1() {
  return new FakeModel();
}

extern "C" void buddy_destroy_audio_transcription_model_v1(
    buddy::runtime::AudioTranscriptionModel *model) {
  delete model;
}

extern "C" const char *buddy_audio_transcription_model_type_v1() {
  return "fake_transcription";
}
