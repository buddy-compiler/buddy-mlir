//===- Qwen3VLResidentModel.h - Resident Qwen3-VL model -------------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_QWEN3VLRESIDENTMODEL_H
#define BUDDY_RUNTIME_MODELS_QWEN3VLRESIDENTMODEL_H

#include "buddy/runtime/core/ResidentModel.h"

#include <memory>
#include <string>

namespace buddy {
namespace runtime {
class Qwen3VLRuntime;

class Qwen3VLResidentModel final : public ResidentModel {
public:
  Qwen3VLResidentModel();
  ~Qwen3VLResidentModel() override;
  Qwen3VLResidentModel(const Qwen3VLResidentModel &) = delete;
  Qwen3VLResidentModel &operator=(const Qwen3VLResidentModel &) = delete;

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

#endif // BUDDY_RUNTIME_MODELS_QWEN3VLRESIDENTMODEL_H
