//===- ResidentModel.h - Long-lived model serving interface ---------------===//
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
//
// Source tree: buddy-mlir/runtime/include/buddy/runtime/core/ResidentModel.h
// Include as:  #include "buddy/runtime/core/ResidentModel.h"
//
// ResidentModel is the boundary between buddy-server transport code and
// model-specific runtime code. The HTTP layer owns JSON/SSE formatting; a
// ResidentModel owns model loading, prompt rendering, tokenization, and
// generation against a long-lived in-memory session.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_CORE_RESIDENTMODEL_H
#define BUDDY_RUNTIME_CORE_RESIDENTMODEL_H

#include "buddy/runtime/core/ServingTypes.h"

#include <string>

namespace buddy {
namespace runtime {

/// Abstract interface for a long-lived in-memory model.
///
/// Implementations are expected to hide model-specific details such as
/// ModelSession, tokenizer choice, chat template rendering, and KV-cache
/// management. Methods may be called from HTTP worker threads; a single-session
/// implementation should serialize generation internally.
class ResidentModel {
public:
  virtual ~ResidentModel() = default;

  ResidentModel(const ResidentModel &) = delete;
  ResidentModel &operator=(const ResidentModel &) = delete;

  /// Load model artifacts and keep weights/session state resident in memory.
  virtual void load(const ResidentModelConfig &cfg) = 0;

  /// Return the current load/runtime status.
  virtual ModelStatus status() const = 0;

  /// Convenience predicate for /health.
  virtual bool isLoaded() const {
    return status().state == ModelLoadState::Ready;
  }

  /// Render chat messages with the model-specific chat template.
  virtual std::string renderChat(const ChatCompletionRequest &request) = 0;

  /// Tokenize text using the model-specific tokenizer.
  virtual TokenizeResult tokenize(const TokenizeRequest &request) = 0;

  /// Run a non-streaming completion on a fully rendered prompt.
  virtual CompletionResult complete(const CompletionRequest &request) = 0;

  /// Run a streaming completion on a fully rendered prompt.
  virtual CompletionResult
  completeStream(const CompletionRequest &request,
                 const CompletionStreamCallback &callback) = 0;

  /// Render chat input and run a non-streaming completion.
  virtual CompletionResult chat(const ChatCompletionRequest &request) = 0;

  /// Render chat input and run a streaming completion.
  virtual CompletionResult
  chatStream(const ChatCompletionRequest &request,
             const CompletionStreamCallback &callback) = 0;

protected:
  ResidentModel() = default;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_RESIDENTMODEL_H
