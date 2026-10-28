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

#include "buddy/runtime/core/ServerModelPluginAdapters.h"

#if BUDDY_FAKE_SERVICE_KIND != 1 && BUDDY_FAKE_SERVICE_KIND != 2 &&            \
    BUDDY_FAKE_SERVICE_KIND != 3
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

  std::string
  renderChat(const buddy::runtime::ChatCompletionRequest &) override {
    return "fake prompt";
  }

  buddy::runtime::TokenizeResult
  tokenize(const buddy::runtime::TokenizeRequest &) override {
    return {{1}, 1};
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

namespace {
#if BUDDY_FAKE_SERVICE_KIND == 1
struct Factory {
  static std::unique_ptr<buddy::runtime::ResidentModel> create() {
    return std::make_unique<FakeModel>();
  }
};
using Adapter = buddy::runtime::server_abi::ResidentServerModelAdapter<Factory>;
#elif BUDDY_FAKE_SERVICE_KIND == 2
struct Factory {
  static std::unique_ptr<buddy::runtime::EmbeddingModel> create() {
    return std::make_unique<FakeModel>();
  }
};
using Adapter =
    buddy::runtime::server_abi::EmbeddingServerModelAdapter<Factory>;
#else
struct Factory {
  static std::unique_ptr<buddy::runtime::MaskedLMModel> create() {
    return std::make_unique<FakeModel>();
  }
};
using Adapter = buddy::runtime::server_abi::MaskedLMServerModelAdapter<Factory>;
#endif
} // namespace

extern "C" const BuddyServerModelDescriptorV1 *
buddy_server_model_descriptor_v1() {
  static const BuddyServerModelDescriptorV1 descriptor = {
      sizeof(descriptor),
      BUDDY_SERVER_MODEL_ABI_VERSION_V1,
#if BUDDY_FAKE_SERVICE_KIND == 1
      BUDDY_SERVER_SERVICE_RESIDENT,
      (1ull << BUDDY_SERVER_OPERATION_COMPLETION) |
          (1ull << BUDDY_SERVER_OPERATION_CHAT) |
          (1ull << BUDDY_SERVER_OPERATION_TOKENIZE),
      BUDDY_SERVER_CAPABILITY_STREAMING,
#elif BUDDY_FAKE_SERVICE_KIND == 2
      BUDDY_SERVER_SERVICE_EMBEDDING,
      (1ull << BUDDY_SERVER_OPERATION_EMBEDDING),
      0,
#else
      BUDDY_SERVER_SERVICE_MASKED_LM,
      (1ull << BUDDY_SERVER_OPERATION_MASKED_LM),
      0,
#endif
      BUDDY_SERVER_MODALITY_TEXT,
#if BUDDY_FAKE_SERVICE_KIND == 1
      BUDDY_SERVER_RESULT_TEXT,
#elif BUDDY_FAKE_SERVICE_KIND == 2
      BUDDY_SERVER_RESULT_EMBEDDING_VECTOR,
#else
      BUDDY_SERVER_RESULT_TOKEN_SCORES,
#endif
      1,
      "fake_model",
      Adapter::ops(),
      0};
  return &descriptor;
}
