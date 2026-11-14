//===- ServerModelErrorFakePlugin.cpp - Adapter error fixture ------------===//
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

#include "buddy/runtime/core/ModelError.h"
#include "buddy/runtime/core/ServerModelPluginAdapters.h"

#include <memory>
#include <new>
#include <stdexcept>
#include <string>

namespace {

class ForcedBadAlloc final : public std::bad_alloc {
public:
  const char *what() const noexcept override {
    return "forced resource_exhausted";
  }
};

[[noreturn]] void throwForPrompt(const std::string &prompt) {
  using buddy::runtime::ModelErrorCode;
  using buddy::runtime::ModelException;

  if (prompt == "invalid_argument")
    throw std::invalid_argument("forced invalid_argument");
  if (prompt == "typed_invalid_argument")
    throw ModelException(ModelErrorCode::InvalidArgument,
                         "forced typed_invalid_argument");
  if (prompt == "not_supported")
    throw ModelException(ModelErrorCode::NotSupported, "forced not_supported");
  if (prompt == "model_not_ready")
    throw ModelException(ModelErrorCode::ModelNotReady,
                         "forced model_not_ready");
  if (prompt == "resource_exhausted")
    throw ForcedBadAlloc();
  if (prompt == "typed_resource_exhausted")
    throw ModelException(ModelErrorCode::ResourceExhausted,
                         "forced typed_resource_exhausted");
  if (prompt == "cancelled")
    throw ModelException(ModelErrorCode::Cancelled, "forced cancelled");
  if (prompt == "deadline_exceeded")
    throw ModelException(ModelErrorCode::DeadlineExceeded,
                         "forced deadline_exceeded");
  if (prompt == "model_error")
    throw std::runtime_error("forced model_error");
  if (prompt == "typed_model_error")
    throw ModelException(ModelErrorCode::ModelError,
                         "forced typed_model_error");
  if (prompt == "internal_error")
    throw ModelException(ModelErrorCode::InternalError,
                         "forced internal_error");
  throw 1;
}

class FakeModel final : public buddy::runtime::ResidentModel {
public:
  void load(const buddy::runtime::ResidentModelConfig &config) override {
    statusValue.state = buddy::runtime::ModelLoadState::Ready;
    statusValue.modelName =
        config.modelName.empty() ? "fake_model" : config.modelName;
    statusValue.backend = "test";
  }

  buddy::runtime::ModelStatus status() const override { return statusValue; }

  std::string
  renderChat(const buddy::runtime::ChatCompletionRequest &request) override {
    return request.input;
  }

  buddy::runtime::TokenizeResult
  tokenize(const buddy::runtime::TokenizeRequest &) override {
    return {};
  }

  buddy::runtime::CompletionResult
  complete(const buddy::runtime::CompletionRequest &request) override {
    throwForPrompt(request.prompt);
  }

  buddy::runtime::CompletionResult completeStream(
      const buddy::runtime::CompletionRequest &request,
      const buddy::runtime::CompletionStreamCallback &callback) override {
    buddy::runtime::CompletionChunk chunk;
    chunk.id = "error-test";
    chunk.model = statusValue.modelName;
    chunk.delta = "partial";
    if (!callback(chunk))
      return {};
    throwForPrompt(request.prompt);
  }

  buddy::runtime::CompletionResult
  chat(const buddy::runtime::ChatCompletionRequest &request) override {
    throwForPrompt(request.input);
  }

  buddy::runtime::CompletionResult chatStream(
      const buddy::runtime::ChatCompletionRequest &request,
      const buddy::runtime::CompletionStreamCallback &callback) override {
    buddy::runtime::CompletionRequest completion;
    completion.prompt = request.input;
    return completeStream(completion, callback);
  }

private:
  buddy::runtime::ModelStatus statusValue;
};

struct Factory {
  static std::unique_ptr<buddy::runtime::ResidentModel> create() {
    return std::make_unique<FakeModel>();
  }
};

using Adapter = buddy::runtime::server_abi::ResidentServerModelAdapter<Factory>;

} // namespace

extern "C" const BuddyServerModelDescriptorV1 *
buddy_server_model_descriptor_v1() {
  static const BuddyServerModelDescriptorV1 descriptor = {
      sizeof(descriptor),
      BUDDY_SERVER_MODEL_ABI_VERSION_V1,
      BUDDY_SERVER_SERVICE_RESIDENT,
      (1ull << BUDDY_SERVER_OPERATION_COMPLETION),
      BUDDY_SERVER_CAPABILITY_STREAMING,
      BUDDY_SERVER_MODALITY_TEXT,
      BUDDY_SERVER_RESULT_TEXT,
      1,
      "fake_model",
      Adapter::ops(),
      0};
  return &descriptor;
}
