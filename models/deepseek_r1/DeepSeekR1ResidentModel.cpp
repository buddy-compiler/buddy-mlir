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

#include "buddy/Core/Container.h"
#include "buddy/LLM/ChatTemplate.h"
#include "buddy/LLM/TextContainer.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/llm/TextGeneration.h"
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

using buddy::Text;

static constexpr int kEosToken = 151643; // <|end▁of▁sentence|>
static constexpr int kEotToken = 151647; // <|EOT|>

FinishReason toFinishReason(const GenerationResult &result) {
  if (result.cancelled)
    return FinishReason::Cancelled;
  if (result.hitTokenLimit)
    return FinishReason::Length;
  return FinishReason::Stop;
}

CompletionTimings toTimings(const GenerationResult &result) {
  CompletionTimings timings;
  timings.prefillMs = result.prefillSecs * 1000.0;
  timings.decodeMs = result.decodeSecs * 1000.0;
  timings.tokensPerSecond = result.decodeSecs > 0.0
                                ? result.generatedTokens / result.decodeSecs
                                : 0.0;
  return timings;
}

CompletionUsage toUsage(int promptTokens, const GenerationResult &result) {
  CompletionUsage usage;
  usage.promptTokens = promptTokens;
  usage.completionTokens = result.generatedTokens;
  usage.totalTokens = usage.promptTokens + usage.completionTokens;
  return usage;
}

std::string nextCompletionId() {
  static std::atomic<unsigned long long> nextId{1};
  return "cmpl-" + std::to_string(nextId.fetch_add(1));
}

std::vector<long long>
mergeStopTokenIds(const std::vector<long long> &base,
                  const std::vector<long long> &requestStopTokenIds) {
  std::vector<long long> merged = base;
  for (long long id : requestStopTokenIds) {
    if (std::find(merged.begin(), merged.end(), id) == merged.end())
      merged.push_back(id);
  }
  return merged;
}

} // namespace

class DeepSeekR1ResidentModel::Impl {
public:
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

  std::string renderChat(const ChatCompletionRequest &request) {
    std::lock_guard<std::mutex> lock(mutex);
    ensureLoadedUnlocked();
    return renderChatUnlocked(request);
  }

  TokenizeResult tokenize(const TokenizeRequest &request) {
    std::lock_guard<std::mutex> lock(mutex);
    ensureLoadedUnlocked();
    return tokenizeUnlocked(request.content, request.countOnly);
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
    CompletionRequest completion;
    {
      std::lock_guard<std::mutex> lock(mutex);
      ensureLoadedUnlocked();
      completion.prompt = renderChatUnlocked(request);
      completion.sampling = request.sampling;
    }
    return completeStream(completion, callback);
  }

private:
  void loadUnlocked(const ResidentModelConfig &cfg) {
    std::vector<std::string> resolvedWeightPaths;

    if (!cfg.raxPath.empty()) {
      ModelManifest manifest;
      session = ModelSession::createFromRax(cfg.raxPath, manifest);
      resolvedWeightPaths = manifest.weightPaths;
      vocabPath = manifest.vocabPath.empty()
                      ? (std::filesystem::path(manifest.soPath).parent_path() /
                         "vocab.txt")
                            .string()
                      : manifest.vocabPath;
      if (!manifest.modelName.empty())
        statusValue.modelName = manifest.modelName;
    } else {
      if (cfg.modelSoPath.empty())
        throw std::runtime_error(
            "DeepSeekR1ResidentModel: modelSoPath is empty");
      if (cfg.weightPaths.empty())
        throw std::runtime_error("DeepSeekR1ResidentModel: no weight paths");

      ModelSession::Config mcfg;
      mcfg.modelSoPath = cfg.modelSoPath;
      session = ModelSession::create(mcfg);
      resolvedWeightPaths = cfg.weightPaths;
      vocabPath = cfg.vocabPath.empty()
                      ? (std::filesystem::path(cfg.modelSoPath).parent_path() /
                         "vocab.txt")
                            .string()
                      : cfg.vocabPath;
    }

    if (vocabPath.empty())
      throw std::runtime_error("DeepSeekR1ResidentModel: vocab path is empty");

    session->loadWeights(resolvedWeightPaths);

    stopTokenIds = {kEosToken, kEotToken};
    if (!cfg.chatTemplatePath.empty()) {
      chatTemplate = std::make_unique<buddy::ChatTemplate>(
          buddy::ChatTemplate::fromFile(cfg.chatTemplatePath));
      for (int id : chatTemplate->stopTokenIds()) {
        const auto value = static_cast<long long>(id);
        if (std::find(stopTokenIds.begin(), stopTokenIds.end(), value) ==
            stopTokenIds.end())
          stopTokenIds.push_back(value);
      }
    } else {
      chatTemplate.reset();
    }

    codec.tokenize = [](Text<size_t, 2> &t, const std::string &vocab) {
      t.tokenizeDeepSeekR1(vocab, BUDDY_DSR1_MAX_TOKEN_LEN);
    };
    codec.detokenize = [](Text<size_t, 2> &t) { return t.revertDeepSeekR1(); };
    codec.maxTokenLen = BUDDY_DSR1_MAX_TOKEN_LEN;

    statusValue.contextLength = BUDDY_DSR1_MAX_TOKEN_LEN;
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

    std::vector<buddy::Message> messages;
    messages.reserve(request.messages.empty() ? 1 : request.messages.size());

    if (!request.messages.empty()) {
      for (const ChatMessage &msg : request.messages)
        messages.push_back({msg.role, msg.content});
    } else {
      messages.push_back({"user", request.input});
    }

    return chatTemplate->apply(messages);
  }

  TokenizeResult tokenizeUnlocked(const std::string &content,
                                  bool countOnly) const {
    Text<size_t, 2> tokens(content);
    tokens.tokenizeDeepSeekR1(vocabPath, BUDDY_DSR1_MAX_TOKEN_LEN);

    TokenizeResult result;
    result.count = tokens.getTokenCnt();
    if (!countOnly) {
      result.tokens.reserve(result.count);
      for (std::size_t i = 0; i < result.count; ++i)
        result.tokens.push_back(static_cast<int>(tokens.getData()[i]));
    }
    return result;
  }

  CompletionResult
  completeStreamUnlocked(const CompletionRequest &request,
                         const CompletionStreamCallback &callback) {
    const std::string id = nextCompletionId();
    const int promptTokens =
        static_cast<int>(tokenizeUnlocked(request.prompt, true).count);
    buddy::Sampler sampler(request.sampling.samplerConfig);
    const std::vector<long long> effectiveStopTokenIds =
        mergeStopTokenIds(stopTokenIds, request.sampling.stopTokenIds);

    GenerationResult generation = runGeneration(
        request.prompt, *session, vocabPath, request.sampling.maxTokens,
        effectiveStopTokenIds, sampler, codec, /*suppress=*/true,
        [&](const GenerationChunk &chunk) {
          if (!callback)
            return true;

          CompletionChunk out;
          out.id = id;
          out.model = statusValue.modelName;
          out.delta = chunk.delta;
          out.tokenId = chunk.tokenId;
          return callback(out);
        });

    CompletionResult result;
    result.id = id;
    result.model = statusValue.modelName;
    result.content = std::move(generation.text);
    result.finishReason = toFinishReason(generation);
    result.usage = toUsage(promptTokens, generation);
    result.timings = toTimings(generation);

    if (callback) {
      CompletionChunk done;
      done.id = result.id;
      done.model = result.model;
      done.done = true;
      done.finishReason = result.finishReason;
      done.usage = result.usage;
      done.timings = result.timings;
      (void)callback(done);
    }

    return result;
  }

  mutable std::mutex mutex;
  bool loaded = false;
  ModelStatus statusValue;
  std::unique_ptr<ModelSession> session;
  std::unique_ptr<buddy::ChatTemplate> chatTemplate;
  std::string vocabPath;
  TextCodec codec;
  std::vector<long long> stopTokenIds;
};

DeepSeekR1ResidentModel::DeepSeekR1ResidentModel()
    : impl(std::make_unique<Impl>()) {}

DeepSeekR1ResidentModel::~DeepSeekR1ResidentModel() = default;

void DeepSeekR1ResidentModel::load(const ResidentModelConfig &cfg) {
  impl->load(cfg);
}

ModelStatus DeepSeekR1ResidentModel::status() const { return impl->status(); }

std::string
DeepSeekR1ResidentModel::renderChat(const ChatCompletionRequest &request) {
  return impl->renderChat(request);
}

TokenizeResult
DeepSeekR1ResidentModel::tokenize(const TokenizeRequest &request) {
  return impl->tokenize(request);
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
