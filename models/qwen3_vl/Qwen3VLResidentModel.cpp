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

#include <atomic>
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

std::string renderMessages(const ChatCompletionRequest &request) {
  if (!request.messages.empty()) {
    std::string prompt;
    for (const ChatMessage &message : request.messages) {
      if (message.content.empty())
        continue;
      if (!prompt.empty())
        prompt += "\n";
      prompt += message.content;
    }
    if (!prompt.empty())
      return prompt;
  }
  if (!request.input.empty())
    return request.input;
  throw std::runtime_error("qwen3_vl: chat request has no text content");
}

std::vector<ImageInput> imagesForRequest(const ChatCompletionRequest &request) {
  std::vector<ImageInput> images = request.images;
  if (images.size() > 1)
    throw std::runtime_error("qwen3_vl: only one image is supported");

  for (const ChatMessage &message : request.messages) {
    if (!message.images.empty() && message.role != "user")
      throw std::runtime_error(
          "qwen3_vl: image content is only supported in user messages");
    for (const ImageInput &image : message.images) {
      if (images.empty())
        images.push_back(image);
      else if (images.front().uri != image.uri ||
               images.front().bytes != image.bytes)
        throw std::runtime_error("qwen3_vl: only one image is supported");
    }
  }
  if (images.size() > 1)
    throw std::runtime_error("qwen3_vl: only one image is supported");
  return images;
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

      ModelStatus ready;
      ready.state = ModelLoadState::Ready;
      ready.backend = "cpu";
      ready.modelName =
          config.modelName.empty() ? candidate->modelName() : config.modelName;
      ready.contextLength = candidate->contextLength();
      ready.message = "model loaded";

      std::lock_guard<std::mutex> lock(mutex);
      runtime = std::move(candidate);
      statusValue = std::move(ready);
    } catch (const std::exception &ex) {
      std::lock_guard<std::mutex> lock(mutex);
      runtime.reset();
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

  std::string renderChat(const ChatCompletionRequest &request) const {
    (void)snapshotRuntime();
    return renderMessages(request);
  }

  TokenizeResult tokenize(const TokenizeRequest &request) const {
    return snapshotRuntime()->tokenize(request.content, request.countOnly);
  }

  CompletionResult complete(const CompletionRequest &request) const {
    return completeStream(request, {});
  }

  CompletionResult
  completeStream(const CompletionRequest &request,
                 const CompletionStreamCallback &callback) const {
    auto model = snapshotRuntime();
    const std::string responseModel = status().modelName;
    CompletionResult result = model->generate(request.prompt, request.images,
                                              request.sampling, callback);
    if (result.id.empty())
      result.id = nextCompletionId();
    result.model = responseModel;
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

  CompletionResult chat(const ChatCompletionRequest &request) const {
    return chatStream(request, {});
  }

  CompletionResult chatStream(const ChatCompletionRequest &request,
                              const CompletionStreamCallback &callback) const {
    CompletionRequest completion;
    completion.prompt = renderMessages(request);
    completion.images = imagesForRequest(request);
    completion.sampling = request.sampling;
    return completeStream(completion, callback);
  }

private:
  std::shared_ptr<Qwen3VLRuntime> snapshotRuntime() const {
    std::lock_guard<std::mutex> lock(mutex);
    if (!runtime || !runtime->isLoaded())
      throw std::runtime_error("qwen3_vl: model is not loaded");
    return runtime;
  }

  mutable std::mutex loadMutex;
  mutable std::mutex mutex;
  std::shared_ptr<Qwen3VLRuntime> runtime;
  ModelStatus statusValue;
};

Qwen3VLResidentModel::Qwen3VLResidentModel() : impl(std::make_unique<Impl>()) {}

Qwen3VLResidentModel::~Qwen3VLResidentModel() = default;

void Qwen3VLResidentModel::load(const ResidentModelConfig &config) {
  impl->load(config);
}

ModelStatus Qwen3VLResidentModel::status() const { return impl->status(); }

std::string
Qwen3VLResidentModel::renderChat(const ChatCompletionRequest &request) {
  return impl->renderChat(request);
}

TokenizeResult Qwen3VLResidentModel::tokenize(const TokenizeRequest &request) {
  return impl->tokenize(request);
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
