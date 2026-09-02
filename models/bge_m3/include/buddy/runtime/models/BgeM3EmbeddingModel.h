//===- BgeM3EmbeddingModel.h - Resident BGE-M3 embedding adapter ----------===//
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

#ifndef BUDDY_RUNTIME_MODELS_BGEM3EMBEDDINGMODEL_H
#define BUDDY_RUNTIME_MODELS_BGEM3EMBEDDINGMODEL_H

#include "buddy/runtime/core/EmbeddingModel.h"

#include <memory>
#include <mutex>

namespace buddy {
namespace runtime {

class BgeM3Runtime;

class BgeM3EmbeddingModel final : public EmbeddingModel {
public:
  BgeM3EmbeddingModel();
  ~BgeM3EmbeddingModel() override;

  void load(const EmbeddingModelConfig &cfg) override;
  ModelStatus status() const override;
  EmbeddingResult embed(const EmbeddingRequest &request) override;

private:
  mutable std::mutex mutex;
  std::unique_ptr<BgeM3Runtime> runtime;
  ModelStatus modelStatus;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_BGEM3EMBEDDINGMODEL_H
