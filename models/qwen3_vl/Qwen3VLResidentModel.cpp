//===- Qwen3VLResidentModel.cpp - Qwen3-VL serving model ------------------===//
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

#include "buddy/runtime/models/Qwen3VLResidentModel.h"

#include "Qwen3VLRuntime.h"

#include "buddy/LLM/ChatTemplate.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {
namespace {

std::string nextCompletionId() {
  static std::atomic<unsigned long long> nextId{1};
  return "qwen-cmpl-" + std::to_string(nextId.fetch_add(1));
}

std::vector<std::string> rolesFor(const buddy::ChatTemplate *tmpl) {
  return tmpl ? tmpl->supportedRoles()
              : std::vector<std::string>{"system", "user", "assistant"};
}

struct RenderedChat {
  std::string prompt;
  std::vector<ImageInput> images;
};

RenderedChat renderMessages(const ChatCompletionRequest &request,
                            const buddy::ChatTemplate *tmpl,
                            std::size_t imageTokenCount) {
  auto source = request.messages;
  if (source.empty()) {
    if (request.input.empty())
      throw std::invalid_argument("qwen3_vl: chat request has no messages");
    source.push_back({"user", request.input, {}});
  }
  const auto roles = rolesFor(tmpl);
  std::size_t selected = source.size();
  std::size_t firstUser = source.size();
  std::size_t imageCount = 0;
  RenderedChat result;
  for (std::size_t i = 0; i < source.size(); ++i) {
    const auto &message = source[i];
    if (std::find(roles.begin(), roles.end(), message.role) == roles.end())
      throw std::invalid_argument("qwen3_vl: unsupported chat role '" +
                                  message.role + "'");
    if (message.role == "user" && firstUser == source.size())
      firstUser = i;
    if (message.imagesAfterText)
      throw std::invalid_argument(
          "qwen3_vl: image content must precede message text");
    if (!message.images.empty()) {
      if (message.role != "user")
        throw std::invalid_argument(
            "qwen3_vl: image content is only supported in user messages");
      imageCount += message.images.size();
      selected = i;
      result.images = message.images;
    }
  }
  if (firstUser == source.size())
    throw std::invalid_argument("qwen3_vl: a user message is required");
  if (imageCount > 1 || request.images.size() > 1)
    throw std::invalid_argument("qwen3_vl: only one image is supported");
  if (imageCount == 1 && !request.images.empty()) {
    // The HTTP decoder mirrors message images into request.images. Accept
    // that mirror, but never collapse two occurrences in the message list.
    const auto &left = result.images.front();
    const auto &right = request.images.front();
    if (left.uri != right.uri || left.bytes != right.bytes ||
        left.mimeType != right.mimeType)
      throw std::invalid_argument("qwen3_vl: conflicting image inputs");
  } else if (imageCount == 0) {
    result.images = request.images;
    selected = firstUser;
  }
  for (const auto &image : result.images) {
    if (!image.bytes.empty() || image.uri.empty() ||
        image.uri.find("http:") == 0 || image.uri.find("https:") == 0 ||
        image.uri.find("data:") == 0)
      throw std::invalid_argument(
          "qwen3_vl: images require one local file URI/path");
  }
  std::string imageBlock = "<|vision_start|>";
  for (std::size_t i = 0; i < imageTokenCount; ++i)
    imageBlock += "<|image_pad|>";
  imageBlock += "<|vision_end|>";
  source[selected].content.insert(0, imageBlock);

  std::vector<buddy::Message> messages;
  for (const auto &message : source)
    messages.push_back({message.role, message.content});
  if (tmpl) {
    if (tmpl->systemInFirstUser()) {
      bool seenSystem = false;
      bool seenNonSystem = false;
      for (const auto &message : messages) {
        if (message.role == "system") {
          if (seenSystem || seenNonSystem)
            throw std::invalid_argument(
                "messages: this template requires one leading system message");
          seenSystem = true;
        } else {
          seenNonSystem = true;
        }
      }
      if (seenSystem && (messages.size() < 2 || messages[1].role != "user"))
        throw std::invalid_argument(
            "messages: this template requires a user immediately after system");
    }
    result.prompt = tmpl->apply(messages);
  } else {
    for (const auto &message : messages)
      result.prompt += "<|im_start|>" + message.role + "\n" + message.content +
                       "<|im_end|>\n";
    result.prompt += "<|im_start|>assistant\n";
  }
  return result;
}

std::shared_ptr<const buddy::ChatTemplate>
loadTemplate(const std::string &path, const Qwen3VLRuntime &runtime) {
  if (path.empty())
    return {};
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    throw std::runtime_error("qwen3_vl: cannot read chat template");
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed)
    throw std::runtime_error("qwen3_vl: cannot parse chat template: " +
                             llvm::toString(parsed.takeError()));
  const auto *object = parsed->getAsObject();
  if (!object)
    throw std::runtime_error("qwen3_vl: chat template must be an object");
  if (const auto *value = object->get("stop_token_ids")) {
    const auto *stops = value->getAsArray();
    if (!stops)
      throw std::invalid_argument("qwen3_vl: invalid template stop_token_ids");
    for (const auto &stop : *stops) {
      auto id = stop.getAsInteger();
      if (!id || *id < 0 || *id > std::numeric_limits<int>::max())
        throw std::invalid_argument(
            "qwen3_vl: invalid template stop_token_ids");
      (void)runtime.detokenize({{static_cast<int>(*id)}});
    }
  }
  if (const auto *value = object->get("stop_tokens")) {
    const auto *stops = value->getAsArray();
    if (!stops)
      throw std::invalid_argument("qwen3_vl: invalid template stop_tokens");
    for (const auto &stop : *stops) {
      auto text = stop.getAsString();
      if (!text || text->empty())
        throw std::invalid_argument("qwen3_vl: invalid template stop_tokens");
    }
  }
  return std::make_shared<const buddy::ChatTemplate>(
      buddy::ChatTemplate::fromJSON(*parsed));
}

} // namespace

class Qwen3VLResidentModel::Impl {
public:
  void load(const ResidentModelConfig &config) {
    std::lock_guard<std::mutex> loadLock(loadMutex);
    {
      std::lock_guard<std::mutex> lock(mutex);
      statusValue.state = ModelLoadState::Loading;
      statusValue.backend = "cpu";
      statusValue.modelName =
          config.modelName.empty() ? "qwen3_vl" : config.modelName;
      statusValue.message = "model is loading";
    }
    try {
      if (config.raxPath.empty())
        throw std::runtime_error(
            "qwen3_vl: --model <path.rax> is required for serving");
      auto candidate = std::make_shared<Qwen3VLRuntime>();
      candidate->load(config.raxPath);
      auto candidateTemplate =
          loadTemplate(config.chatTemplatePath, *candidate);
      ModelStatus ready;
      ready.state = ModelLoadState::Ready;
      ready.backend = "cpu";
      ready.modelName =
          config.modelName.empty() ? candidate->modelName() : config.modelName;
      ready.contextLength = candidate->contextLength();
      ready.message = "model loaded";
      std::lock_guard<std::mutex> lock(mutex);
      runtime = std::move(candidate);
      chatTemplate = std::move(candidateTemplate);
      statusValue = std::move(ready);
    } catch (const std::exception &ex) {
      std::lock_guard<std::mutex> lock(mutex);
      runtime.reset();
      chatTemplate.reset();
      statusValue.state = ModelLoadState::Error;
      statusValue.backend = "cpu";
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
    if (statusValue.state != ModelLoadState::Ready || !runtime)
      return {};
    auto result = runtime->capabilities();
    if (result.maxNewTokens) {
      result.chatRoles = rolesFor(chatTemplate.get());
      result.chatMultiTurn = !result.chatRoles.empty();
    }
    return result;
  }

  std::string renderChat(const ChatCompletionRequest &request) const {
    auto state = snapshot();
    return renderMessages(request, state.tmpl.get(),
                          state.runtime->imageTokenCount())
        .prompt;
  }

  TokenizeResult tokenize(const TokenizeRequest &request) const {
    return snapshot().runtime->tokenize(request);
  }

  DetokenizeResult detokenize(const DetokenizeRequest &request) const {
    return snapshot().runtime->detokenize(request);
  }

  CompletionResult complete(const CompletionRequest &request) const {
    return completeStream(request, {});
  }

  CompletionResult
  completeStream(const CompletionRequest &request,
                 const CompletionStreamCallback &callback) const {
    return generate(snapshot(), request.prompt, request.images,
                    request.sampling, callback);
  }

  CompletionResult chat(const ChatCompletionRequest &request) const {
    return chatStream(request, {});
  }

  CompletionResult chatStream(const ChatCompletionRequest &request,
                              const CompletionStreamCallback &callback) const {
    auto state = snapshot();
    auto rendered = renderMessages(request, state.tmpl.get(),
                                   state.runtime->imageTokenCount());
    return generate(state, rendered.prompt, rendered.images, request.sampling,
                    callback);
  }

private:
  struct Snapshot {
    std::shared_ptr<Qwen3VLRuntime> runtime;
    std::shared_ptr<const buddy::ChatTemplate> tmpl;
    std::string model;
  };

  Snapshot snapshot() const {
    std::lock_guard<std::mutex> lock(mutex);
    if (statusValue.state != ModelLoadState::Ready || !runtime ||
        !runtime->isLoaded())
      throw std::runtime_error("qwen3_vl: model is not loaded");
    return {runtime, chatTemplate, statusValue.modelName};
  }

  CompletionResult generate(const Snapshot &state, const std::string &prompt,
                            const std::vector<ImageInput> &images,
                            const SamplingParams &sampling,
                            const CompletionStreamCallback &callback) const {
    auto params = sampling;
    if (state.tmpl) {
      for (int token : state.tmpl->stopTokenIds())
        params.stopTokenIds.push_back(token);
      for (const auto &text : state.tmpl->stopTokens())
        params.stop.push_back(text);
    }
    const std::string id = nextCompletionId();
    CompletionStreamCallback wrapped;
    if (callback)
      wrapped = [&](const CompletionChunk &chunk) {
        auto event = chunk;
        event.id = id;
        event.model = state.model;
        return callback(event);
      };
    auto result =
        state.runtime->generateServing(prompt, images, params, wrapped);
    result.id = id;
    result.model = state.model;
    return result;
  }

  mutable std::mutex loadMutex;
  mutable std::mutex mutex;
  std::shared_ptr<Qwen3VLRuntime> runtime;
  std::shared_ptr<const buddy::ChatTemplate> chatTemplate;
  ModelStatus statusValue;
};

Qwen3VLResidentModel::Qwen3VLResidentModel() : impl(std::make_unique<Impl>()) {}

Qwen3VLResidentModel::~Qwen3VLResidentModel() = default;

void Qwen3VLResidentModel::load(const ResidentModelConfig &config) {
  impl->load(config);
}

ModelStatus Qwen3VLResidentModel::status() const { return impl->status(); }

ResidentModelCapabilities Qwen3VLResidentModel::capabilities() const {
  return impl->capabilities();
}

std::string
Qwen3VLResidentModel::renderChat(const ChatCompletionRequest &request) {
  return impl->renderChat(request);
}

TokenizeResult Qwen3VLResidentModel::tokenize(const TokenizeRequest &request) {
  return impl->tokenize(request);
}

DetokenizeResult
Qwen3VLResidentModel::detokenize(const DetokenizeRequest &request) {
  return impl->detokenize(request);
}

CompletionResult
Qwen3VLResidentModel::complete(const CompletionRequest &request) {
  return impl->complete(request);
}

CompletionResult
Qwen3VLResidentModel::completeStream(const CompletionRequest &request,
                                     const CompletionStreamCallback &callback) {
  return impl->completeStream(request, callback);
}

CompletionResult
Qwen3VLResidentModel::chat(const ChatCompletionRequest &request) {
  return impl->chat(request);
}

CompletionResult
Qwen3VLResidentModel::chatStream(const ChatCompletionRequest &request,
                                 const CompletionStreamCallback &callback) {
  return impl->chatStream(request, callback);
}

} // namespace runtime
} // namespace buddy
