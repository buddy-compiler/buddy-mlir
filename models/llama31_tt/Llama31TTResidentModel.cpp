//===- Llama31TTResidentModel.cpp - Llama 3.1 TT resident model ----------===//
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

#include "buddy/runtime/models/Llama31TTResidentModel.h"

#include <mutex>
#include <stdexcept>
#include <utility>

namespace buddy {
namespace runtime {

class Llama31TTResidentModel::Impl {
public:
  explicit Impl(std::shared_ptr<Llama31TTExecution> value)
      : execution(std::move(value)) {
    if (!execution)
      execution = std::make_shared<Llama31TTExecution>();
    statusValue.modelName = "llama31_tt";
    statusValue.backend = "ttnn";
  }

  void load(const ResidentModelConfig &config) {
    std::lock_guard<std::mutex> lock(mutex);
    statusValue.state = ModelLoadState::Loading;
    statusValue.message = "model is loading";
    try {
      execution->load(config);
      statusValue.state = ModelLoadState::Ready;
      statusValue.modelName = execution->metadata().modelName;
      statusValue.contextLength = execution->metadata().maxCacheLen;
      statusValue.message = "model loaded";
    } catch (const std::exception &error) {
      statusValue.state = ModelLoadState::Error;
      statusValue.message = error.what();
      throw;
    }
  }

  ModelStatus status() const {
    std::lock_guard<std::mutex> lock(mutex);
    return statusValue;
  }

  std::string renderChat(const ChatCompletionRequest &request) {
    std::lock_guard<std::mutex> lock(mutex);
    ensureReady();
    validateModel(request.model);
    return execution->renderChat(request);
  }

  TokenizeResult tokenize(const TokenizeRequest &request) {
    std::lock_guard<std::mutex> lock(mutex);
    ensureReady();
    if (request.content.empty())
      throw std::invalid_argument("llama31_tt: tokenize content is empty");
    return execution->tokenize(request.content, request.countOnly,
                               request.addSpecial);
  }

  CompletionResult complete(const CompletionRequest &request) {
    return completeStream(request, {});
  }

  CompletionResult completeStream(const CompletionRequest &request,
                                  const CompletionStreamCallback &callback) {
    std::lock_guard<std::mutex> lock(mutex);
    ensureReady();
    if (!request.images.empty())
      throw std::invalid_argument("llama31_tt: image inputs are not supported");
    CompletionResult result = execution->generate(
        request.prompt, request.sampling, [&](const CompletionChunk &source) {
          if (!callback)
            return true;
          CompletionChunk chunk = source;
          chunk.model = statusValue.modelName;
          return callback(chunk);
        });
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

  CompletionResult chat(const ChatCompletionRequest &request) {
    return chatStream(request, {});
  }

  CompletionResult chatStream(const ChatCompletionRequest &request,
                              const CompletionStreamCallback &callback) {
    std::lock_guard<std::mutex> lock(mutex);
    ensureReady();
    validateModel(request.model);
    CompletionRequest completion;
    completion.prompt = execution->renderChat(request);
    completion.sampling = request.sampling;
    CompletionResult result =
        execution->generate(completion.prompt, completion.sampling,
                            [&](const CompletionChunk &source) {
                              if (!callback)
                                return true;
                              CompletionChunk chunk = source;
                              chunk.model = statusValue.modelName;
                              return callback(chunk);
                            });
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

private:
  void ensureReady() const {
    if (statusValue.state != ModelLoadState::Ready || !execution->isLoaded())
      throw std::runtime_error("llama31_tt: model is not loaded");
  }

  void validateModel(const std::string &requested) const {
    if (!requested.empty() && requested != statusValue.modelName)
      throw std::invalid_argument(
          "llama31_tt: requested model does not match loaded model '" +
          statusValue.modelName + "'");
  }

  mutable std::mutex mutex;
  std::shared_ptr<Llama31TTExecution> execution;
  ModelStatus statusValue;
};

Llama31TTResidentModel::Llama31TTResidentModel()
    : impl(std::make_unique<Impl>(nullptr)) {}

Llama31TTResidentModel::Llama31TTResidentModel(
    std::shared_ptr<Llama31TTExecution> execution)
    : impl(std::make_unique<Impl>(std::move(execution))) {}

Llama31TTResidentModel::~Llama31TTResidentModel() = default;

void Llama31TTResidentModel::load(const ResidentModelConfig &cfg) {
  impl->load(cfg);
}
ModelStatus Llama31TTResidentModel::status() const { return impl->status(); }
std::string
Llama31TTResidentModel::renderChat(const ChatCompletionRequest &request) {
  return impl->renderChat(request);
}
TokenizeResult
Llama31TTResidentModel::tokenize(const TokenizeRequest &request) {
  return impl->tokenize(request);
}
CompletionResult
Llama31TTResidentModel::complete(const CompletionRequest &request) {
  return impl->complete(request);
}
CompletionResult Llama31TTResidentModel::completeStream(
    const CompletionRequest &request,
    const CompletionStreamCallback &callback) {
  return impl->completeStream(request, callback);
}
CompletionResult
Llama31TTResidentModel::chat(const ChatCompletionRequest &request) {
  return impl->chat(request);
}
CompletionResult
Llama31TTResidentModel::chatStream(const ChatCompletionRequest &request,
                                   const CompletionStreamCallback &callback) {
  return impl->chatStream(request, callback);
}

} // namespace runtime
} // namespace buddy
