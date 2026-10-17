//===- DeepSeekR1ResidentModel.cpp - DeepSeekR1 serving model -------------===//
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

#include "buddy/runtime/models/DeepSeekR1ResidentModel.h"

#include "buddy/LLM/ChatTemplate.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/llm/ServingGeneration.h"
#include "buddy/runtime/models/DeepSeekR1RawCodec.h"
#include "buddy/runtime/models/ModelSession.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace buddy {
namespace runtime {

namespace {

std::string nextCompletionId() {
  static std::atomic<unsigned long long> nextId{1};
  return "cmpl-" + std::to_string(nextId.fetch_add(1));
}

DeepSeekR1ResidentModel::SessionResources
createProductionSession(const ResidentModelConfig &cfg) {
  DeepSeekR1ResidentModel::SessionResources resources;
  std::string modelSoPath = cfg.modelSoPath;
  if (!cfg.raxPath.empty()) {
    ModelManifest manifest;
    resources.session = ModelSession::createFromRax(cfg.raxPath, manifest);
    resources.weightPaths = manifest.weightPaths;
    resources.vocabPath = manifest.vocabPath;
    resources.modelName = manifest.modelName;
    modelSoPath = manifest.soPath;
  } else {
    if (cfg.modelSoPath.empty())
      throw std::runtime_error("DeepSeekR1ResidentModel: modelSoPath is empty");
    if (cfg.weightPaths.empty())
      throw std::runtime_error("DeepSeekR1ResidentModel: no weight paths");
    ModelSession::Config modelConfig;
    modelConfig.modelSoPath = cfg.modelSoPath;
    resources.session = ModelSession::create(modelConfig);
    resources.weightPaths = cfg.weightPaths;
    resources.vocabPath = cfg.vocabPath;
  }
  if (resources.vocabPath.empty())
    resources.vocabPath =
        (std::filesystem::path(modelSoPath).parent_path() / "vocab.txt")
            .string();
  resources.contextLength = BUDDY_DSR1_MAX_TOKEN_LEN;
  return resources;
}

} // namespace

class DeepSeekR1ResidentModel::Impl {
public:
  explicit Impl(SessionFactory factory) : sessionFactory(std::move(factory)) {
    if (!sessionFactory)
      throw std::invalid_argument("deepseek_r1: session factory is empty");
  }

  void load(const ResidentModelConfig &cfg) {
    std::lock_guard<std::mutex> lock(mutex);

    statusValue = {};
    statusValue.state = ModelLoadState::Loading;
    statusValue.modelName =
        cfg.modelName.empty() ? "deepseek_r1" : cfg.modelName;
    statusValue.backend = "cpu";

    try {
      loadUnlocked(cfg);
      statusValue.state = ModelLoadState::Ready;
      statusValue.message = "model loaded";
    } catch (const std::exception &ex) {
      session.reset();
      chatTemplate.reset();
      rawCodec.reset();
      loaded = false;
      statusValue.state = ModelLoadState::Error;
      statusValue.message = ex.what();
      throw;
    }
  }

  ModelStatus status() const {
    std::lock_guard<std::mutex> lock(mutex);
    return statusValue;
  }

  ResidentModelCapabilities capabilities() const {
    std::lock_guard<std::mutex> lock(mutex);
    ResidentModelCapabilities result;
    if (!loaded)
      return result;
    result.rawTokenize = true;
    result.rawDetokenize = true;
    result.tokenizeAddSpecial = rawCodec->tokenId("<|begin▁of▁sentence|>") >= 0;
    result.tokenizeParseSpecial = true;
    if (chatTemplate) {
      result.chatRoles = chatTemplate->supportedRoles();
      result.chatMultiTurn = true;
    }
    result.streaming = true;
    result.maxNewTokens = true;
    result.stop = true;
    result.stopTokenIds = true;
    result.sampling.temperature = true;
    result.sampling.topK = true;
    result.sampling.topP = true;
    result.sampling.minP = true;
    result.sampling.repeatPenalty = true;
    result.sampling.repeatLastN = true;
    result.sampling.seed = true;
    return result;
  }

  std::string renderChat(const ChatCompletionRequest &request) {
    std::lock_guard<std::mutex> lock(mutex);
    ensureLoadedUnlocked();
    return renderChatUnlocked(request);
  }

  TokenizeResult tokenize(const TokenizeRequest &request) {
    std::lock_guard<std::mutex> lock(mutex);
    ensureLoadedUnlocked();
    return rawCodec->tokenize(request);
  }

  DetokenizeResult detokenize(const DetokenizeRequest &request) {
    std::lock_guard<std::mutex> lock(mutex);
    ensureLoadedUnlocked();
    return rawCodec->detokenize(request);
  }

  CompletionResult complete(const CompletionRequest &request) {
    return completeStream(request, nullptr);
  }

  CompletionResult completeStream(const CompletionRequest &request,
                                  const CompletionStreamCallback &callback) {
    std::lock_guard<std::mutex> lock(mutex);
    ensureLoadedUnlocked();
    return completeStreamUnlocked(request, callback);
  }

  CompletionResult chat(const ChatCompletionRequest &request) {
    return chatStream(request, nullptr);
  }

  CompletionResult chatStream(const ChatCompletionRequest &request,
                              const CompletionStreamCallback &callback) {
    std::lock_guard<std::mutex> lock(mutex);
    ensureLoadedUnlocked();
    CompletionRequest completion;
    completion.prompt = renderChatUnlocked(request);
    completion.sampling = request.sampling;
    // The configured template decides whether chat needs a BOS. Encoding the
    // rendered prompt must not add any second layer of model scaffolding.
    return completeStreamUnlocked(completion, callback, false);
  }

private:
  void loadUnlocked(const ResidentModelConfig &cfg) {
    loaded = false;
    auto resources = sessionFactory(cfg);
    if (!resources.session || resources.contextLength <= 0)
      throw std::runtime_error("deepseek_r1: invalid session resources");
    session = std::move(resources.session);
    if (resources.vocabPath.empty())
      throw std::runtime_error("DeepSeekR1ResidentModel: vocab path is empty");
    rawCodec = std::make_unique<DeepSeekR1RawCodec>(resources.vocabPath);
    if (rawCodec->vocabSize() > session->vocabSize())
      throw std::runtime_error(
          "deepseek_r1: vocabulary exceeds session logits");
    session->loadWeights(resources.weightPaths);
    if (cfg.modelName.empty() && !resources.modelName.empty())
      statusValue.modelName = resources.modelName;

    stopTokenIds.clear();
    for (const char *piece : {"<|end▁of▁sentence|>", "<|EOT|>"}) {
      const int id = rawCodec->tokenId(piece);
      if (id >= 0)
        stopTokenIds.push_back(id);
    }
    defaultStopStrings.clear();
    if (!cfg.chatTemplatePath.empty()) {
      chatTemplate = std::make_unique<buddy::ChatTemplate>(
          buddy::ChatTemplate::fromFile(cfg.chatTemplatePath));
      for (int id : chatTemplate->stopTokenIds()) {
        (void)rawCodec->tokenBytes(id);
        const auto value = static_cast<long long>(id);
        if (std::find(stopTokenIds.begin(), stopTokenIds.end(), value) ==
            stopTokenIds.end())
          stopTokenIds.push_back(value);
      }
      defaultStopStrings = chatTemplate->stopTokens();
    } else {
      chatTemplate.reset();
    }

    statusValue.contextLength = resources.contextLength;
    loaded = true;
  }

  void ensureLoadedUnlocked() const {
    if (!loaded || !session)
      throw std::runtime_error("DeepSeekR1ResidentModel: model is not loaded");
  }

  std::string renderChatUnlocked(const ChatCompletionRequest &request) const {
    if (!chatTemplate)
      throw std::runtime_error(
          "DeepSeekR1ResidentModel: chat template is not configured");

    if (!request.images.empty())
      throw std::invalid_argument(
          "image_path: DeepSeek does not support images");
    const auto supportedRoles = chatTemplate->supportedRoles();

    std::vector<buddy::Message> messages;
    messages.reserve(request.messages.empty() ? 1 : request.messages.size());

    if (!request.messages.empty()) {
      for (std::size_t index = 0; index < request.messages.size(); ++index) {
        const ChatMessage &msg = request.messages[index];
        if (!msg.images.empty())
          throw std::invalid_argument(
              "messages: DeepSeek does not support images");
        if (std::find(supportedRoles.begin(), supportedRoles.end(), msg.role) ==
            supportedRoles.end())
          throw std::invalid_argument("messages.role: unsupported chat role '" +
                                      msg.role + "'");
        if (msg.role == "system" && chatTemplate->systemInFirstUser() &&
            (index != 0 || request.messages.size() < 2 ||
             request.messages[1].role != "user"))
          throw std::invalid_argument(
              "messages: this template requires one leading system message "
              "followed by user");
        messages.push_back({msg.role, msg.content});
      }
    } else {
      if (std::find(supportedRoles.begin(), supportedRoles.end(), "user") ==
          supportedRoles.end())
        throw std::invalid_argument(
            "messages.role: template does not support user");
      messages.push_back({"user", request.input});
    }

    return chatTemplate->apply(messages);
  }

  CompletionResult
  completeStreamUnlocked(const CompletionRequest &request,
                         const CompletionStreamCallback &callback,
                         bool addSpecial = true) {
    const std::string id = nextCompletionId();
    if (!request.images.empty())
      throw std::invalid_argument(
          "image_path: DeepSeek does not support images");
    SamplingParams sampling = request.sampling;
    for (long long stop : stopTokenIds)
      if (std::find(sampling.stopTokenIds.begin(), sampling.stopTokenIds.end(),
                    stop) == sampling.stopTokenIds.end())
        sampling.stopTokenIds.push_back(stop);
    for (long long stop : sampling.stopTokenIds) {
      if (stop < 0 || stop >= rawCodec->vocabSize())
        throw std::invalid_argument(
            "stop_token_ids: token ID is outside the vocabulary");
    }
    sampling.stop.insert(sampling.stop.end(), defaultStopStrings.begin(),
                         defaultStopStrings.end());
    ServingTextCodec codec;
    codec.contextLength = statusValue.contextLength;
    codec.vocabSize = rawCodec->vocabSize();
    codec.encode = [&](const std::string &prompt) {
      return rawCodec->encode(prompt, addSpecial, true);
    };
    codec.tokenBytes = [&](int token) { return rawCodec->tokenBytes(token); };
    CompletionResult result =
        runServingGeneration(request.prompt, *session, sampling, codec,
                             [&](const CompletionChunk &chunk) {
                               if (!callback)
                                 return true;
                               CompletionChunk out = chunk;
                               out.id = id;
                               out.model = statusValue.modelName;
                               return callback(out);
                             });
    result.id = id;
    result.model = statusValue.modelName;

    return result;
  }

  mutable std::mutex mutex;
  bool loaded = false;
  ModelStatus statusValue;
  SessionFactory sessionFactory;
  std::unique_ptr<LLMSession> session;
  std::unique_ptr<buddy::ChatTemplate> chatTemplate;
  std::unique_ptr<DeepSeekR1RawCodec> rawCodec;
  std::vector<long long> stopTokenIds;
  std::vector<std::string> defaultStopStrings;
};

DeepSeekR1ResidentModel::DeepSeekR1ResidentModel()
    : DeepSeekR1ResidentModel(createProductionSession) {}

DeepSeekR1ResidentModel::DeepSeekR1ResidentModel(SessionFactory sessionFactory)
    : impl(std::make_unique<Impl>(std::move(sessionFactory))) {}

DeepSeekR1ResidentModel::~DeepSeekR1ResidentModel() = default;

void DeepSeekR1ResidentModel::load(const ResidentModelConfig &cfg) {
  impl->load(cfg);
}

ModelStatus DeepSeekR1ResidentModel::status() const { return impl->status(); }

ResidentModelCapabilities DeepSeekR1ResidentModel::capabilities() const {
  return impl->capabilities();
}

std::string
DeepSeekR1ResidentModel::renderChat(const ChatCompletionRequest &request) {
  return impl->renderChat(request);
}

TokenizeResult
DeepSeekR1ResidentModel::tokenize(const TokenizeRequest &request) {
  return impl->tokenize(request);
}

DetokenizeResult
DeepSeekR1ResidentModel::detokenize(const DetokenizeRequest &request) {
  return impl->detokenize(request);
}

CompletionResult
DeepSeekR1ResidentModel::complete(const CompletionRequest &request) {
  return impl->complete(request);
}

CompletionResult DeepSeekR1ResidentModel::completeStream(
    const CompletionRequest &request,
    const CompletionStreamCallback &callback) {
  return impl->completeStream(request, callback);
}

CompletionResult
DeepSeekR1ResidentModel::chat(const ChatCompletionRequest &request) {
  return impl->chat(request);
}

CompletionResult
DeepSeekR1ResidentModel::chatStream(const ChatCompletionRequest &request,
                                    const CompletionStreamCallback &callback) {
  return impl->chatStream(request, callback);
}

} // namespace runtime
} // namespace buddy
