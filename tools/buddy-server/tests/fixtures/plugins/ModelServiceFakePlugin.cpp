//===- ModelServiceFakePlugin.cpp - Fake model service plugins -----------===//
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

#include "buddy/runtime/core/ModelPlugin.h"

#include <stdexcept>

#if BUDDY_FAKE_SERVICE_KIND == 1
#include "buddy/runtime/core/ResidentModelPlugin.h"
#elif BUDDY_FAKE_SERVICE_KIND == 2
#include "buddy/runtime/core/EmbeddingModelPlugin.h"
#elif BUDDY_FAKE_SERVICE_KIND == 3
#include "buddy/runtime/core/MaskedLMModelPlugin.h"
#else
#error "unsupported BUDDY_FAKE_SERVICE_KIND"
#endif

namespace {

buddy::runtime::ModelStatus readyStatus(const std::string &modelName) {
  buddy::runtime::ModelStatus status;
  status.state = buddy::runtime::ModelLoadState::Ready;
  status.modelName = modelName.empty() ? "fake_model" : modelName;
  status.backend = "cpu";
  status.message = "model loaded";
  status.contextLength = 32;
  return status;
}

#if BUDDY_FAKE_SERVICE_KIND == 1
class FakeModel final : public buddy::runtime::ResidentModel {
public:
  void load(const buddy::runtime::ResidentModelConfig &config) override {
    statusValue = readyStatus(config.modelName);
  }

  buddy::runtime::ModelStatus status() const override { return statusValue; }

  buddy::runtime::ResidentModelCapabilities capabilities() const override {
    buddy::runtime::ResidentModelCapabilities caps;
    caps.maxNewTokens = true;
    caps.streaming = true;
    caps.chatRoles = {"system", "user", "assistant"};
    caps.chatMultiTurn = true;
    return caps;
  }

  buddy::runtime::DetokenizeResult
  detokenize(const buddy::runtime::DetokenizeRequest &) override {
    throw std::invalid_argument("fake_model: raw detokenize is not supported");
  }

  std::string
  renderChat(const buddy::runtime::ChatCompletionRequest &) override {
    return "fake prompt";
  }

  buddy::runtime::TokenizeResult
  tokenize(const buddy::runtime::TokenizeRequest &) override {
    return {{1}, 1, {}};
  }

  buddy::runtime::CompletionResult
  complete(const buddy::runtime::CompletionRequest &) override {
    return result();
  }

  buddy::runtime::CompletionResult completeStream(
      const buddy::runtime::CompletionRequest &,
      const buddy::runtime::CompletionStreamCallback &callback) override {
    buddy::runtime::CompletionChunk chunk;
    chunk.id = "fake-completion";
    chunk.model = statusValue.modelName;
    chunk.delta = "fake completion";
    chunk.done = true;
    chunk.finishReason = buddy::runtime::FinishReason::Stop;
    (void)callback(chunk);
    return result();
  }

  buddy::runtime::CompletionResult
  chat(const buddy::runtime::ChatCompletionRequest &) override {
    return result();
  }

  buddy::runtime::CompletionResult chatStream(
      const buddy::runtime::ChatCompletionRequest &,
      const buddy::runtime::CompletionStreamCallback &callback) override {
    buddy::runtime::CompletionRequest request;
    return completeStream(request, callback);
  }

private:
  buddy::runtime::CompletionResult result() const {
    buddy::runtime::CompletionResult value;
    value.id = "fake-completion";
    value.model = statusValue.modelName;
    value.content = "fake completion";
    value.finishReason = buddy::runtime::FinishReason::Stop;
    value.usage = {1, 1, 2};
    return value;
  }

  buddy::runtime::ModelStatus statusValue;
};
#elif BUDDY_FAKE_SERVICE_KIND == 2
class FakeModel final : public buddy::runtime::EmbeddingModel {
public:
  void load(const buddy::runtime::EmbeddingModelConfig &config) override {
    statusValue = readyStatus(config.modelName);
  }

  buddy::runtime::ModelStatus status() const override { return statusValue; }

  buddy::runtime::EmbeddingResult
  embed(const buddy::runtime::EmbeddingRequest &) override {
    return {statusValue.modelName, {0.25F, 0.75F}, 1, 1};
  }

private:
  buddy::runtime::ModelStatus statusValue;
};
#elif BUDDY_FAKE_SERVICE_KIND == 3
class FakeModel final : public buddy::runtime::MaskedLMModel {
public:
  void load(const buddy::runtime::MaskedLMModelConfig &config) override {
    statusValue = readyStatus(config.modelName);
  }

  buddy::runtime::ModelStatus status() const override { return statusValue; }

  buddy::runtime::MaskedLMResult
  predict(const buddy::runtime::MaskedLMRequest &) override {
    buddy::runtime::MaskedLMResult result;
    result.model = statusValue.modelName;
    result.sequenceLength = 1;
    result.promptTokens = 1;
    result.predictions.push_back({0, {{7, "fake", 1.0F}}});
    return result;
  }

private:
  buddy::runtime::ModelStatus statusValue;
};
#endif

} // namespace

#if BUDDY_FAKE_SERVICE_KIND == 1
extern "C" buddy::runtime::ResidentModel *buddy_create_resident_model_v2() {
  return new FakeModel();
}

extern "C" void
buddy_destroy_resident_model_v2(buddy::runtime::ResidentModel *model) {
  delete model;
}

extern "C" const char *buddy_resident_model_type_v2() { return "fake_model"; }
#elif BUDDY_FAKE_SERVICE_KIND == 2
extern "C" buddy::runtime::EmbeddingModel *buddy_create_embedding_model_v1() {
  return new FakeModel();
}

extern "C" void
buddy_destroy_embedding_model_v1(buddy::runtime::EmbeddingModel *model) {
  delete model;
}

extern "C" const char *buddy_embedding_model_type_v1() { return "fake_model"; }
#elif BUDDY_FAKE_SERVICE_KIND == 3
extern "C" buddy::runtime::MaskedLMModel *buddy_create_masked_lm_model_v1() {
  return new FakeModel();
}

extern "C" void
buddy_destroy_masked_lm_model_v1(buddy::runtime::MaskedLMModel *model) {
  delete model;
}

extern "C" const char *buddy_masked_lm_model_type_v1() { return "fake_model"; }
#endif

extern "C" const buddy::runtime::ModelPluginDescriptorV1 *
buddy_model_plugin_descriptor_v1() {
  static constexpr buddy::runtime::ModelPluginDescriptorV1 descriptor = {
      sizeof(descriptor), buddy::runtime::ModelPluginDescriptorAbiVersionV1,
      static_cast<buddy::runtime::ModelServiceKind>(BUDDY_FAKE_SERVICE_KIND),
      0};
  return &descriptor;
}
