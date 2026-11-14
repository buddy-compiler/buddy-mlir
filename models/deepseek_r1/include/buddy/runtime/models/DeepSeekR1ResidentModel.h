//===- DeepSeekR1ResidentModel.h - DeepSeekR1 serving model ---------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_DEEPSEEKR1RESIDENTMODEL_H
#define BUDDY_RUNTIME_MODELS_DEEPSEEKR1RESIDENTMODEL_H

#include "buddy/runtime/core/ResidentModel.h"

#include <memory>

namespace buddy {
namespace runtime {

/// Long-lived DeepSeekR1 model for buddy-server.
///
/// This implementation owns a single ModelSession and keeps loaded weights in
/// memory after load(). Generation is serialized internally, which makes the
/// MVP safe for one resident session while leaving room for a future slot pool.
class DeepSeekR1ResidentModel : public ResidentModel {
public:
  DeepSeekR1ResidentModel();
  ~DeepSeekR1ResidentModel() override;

  DeepSeekR1ResidentModel(const DeepSeekR1ResidentModel &) = delete;
  DeepSeekR1ResidentModel &operator=(const DeepSeekR1ResidentModel &) = delete;

  void load(const ResidentModelConfig &cfg) override;
  ModelStatus status() const override;
  std::string renderChat(const ChatCompletionRequest &request) override;
  TokenizeResult tokenize(const TokenizeRequest &request) override;
  CompletionResult complete(const CompletionRequest &request) override;
  CompletionResult
  completeStream(const CompletionRequest &request,
                 const CompletionStreamCallback &callback) override;
  CompletionResult chat(const ChatCompletionRequest &request) override;
  CompletionResult
  chatStream(const ChatCompletionRequest &request,
             const CompletionStreamCallback &callback) override;

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_DEEPSEEKR1RESIDENTMODEL_H
