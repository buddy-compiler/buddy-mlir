//===- Llama31TTResidentModel.h - Llama 3.1 TT resident model ------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_LLAMA31TTRESIDENTMODEL_H
#define BUDDY_RUNTIME_MODELS_LLAMA31TTRESIDENTMODEL_H

#include "buddy/runtime/core/ResidentModel.h"
#include "buddy/runtime/models/Llama31TTExecution.h"

#include <memory>

namespace buddy {
namespace runtime {

class Llama31TTResidentModel final : public ResidentModel {
public:
  Llama31TTResidentModel();
  explicit Llama31TTResidentModel(
      std::shared_ptr<Llama31TTExecution> execution);
  ~Llama31TTResidentModel() override;
  Llama31TTResidentModel(const Llama31TTResidentModel &) = delete;
  Llama31TTResidentModel &operator=(const Llama31TTResidentModel &) = delete;

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

#endif // BUDDY_RUNTIME_MODELS_LLAMA31TTRESIDENTMODEL_H
