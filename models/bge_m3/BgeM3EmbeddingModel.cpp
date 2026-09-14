//===- BgeM3EmbeddingModel.cpp - Resident BGE-M3 embedding adapter --------===//
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

#include "buddy/runtime/models/BgeM3EmbeddingModel.h"
#include "buddy/runtime/models/BgeM3Runtime.h"

#include <stdexcept>

namespace buddy {
namespace runtime {
BgeM3EmbeddingModel::BgeM3EmbeddingModel() {
  modelStatus.backend = "cpu";
  modelStatus.modelName = "bge_m3";
}
BgeM3EmbeddingModel::~BgeM3EmbeddingModel() = default;

void BgeM3EmbeddingModel::load(const EmbeddingModelConfig &cfg) {
  std::lock_guard<std::mutex> lock(mutex);
  modelStatus = ModelStatus{};
  modelStatus.state = ModelLoadState::Loading;
  modelStatus.backend = "cpu";
  modelStatus.modelName = cfg.modelName.empty() ? "bge_m3" : cfg.modelName;
  try {
    if (!cfg.raxPath.empty()) {
      runtime = BgeM3Runtime::loadFromRax(cfg.raxPath);
    } else {
      if (cfg.modelSoPath.empty())
        throw std::runtime_error(
            "BgeM3EmbeddingModel: model shared library path is empty");
      if (cfg.weightPaths.empty())
        throw std::runtime_error("BgeM3EmbeddingModel: no weights specified");
      if (cfg.vocabPath.empty())
        throw std::runtime_error(
            "BgeM3EmbeddingModel: tokenizer path is empty");
      if (cfg.weightPaths.size() != 1)
        throw std::runtime_error("BgeM3EmbeddingModel: legacy mode accepts "
                                 "exactly one weights file");
      runtime = BgeM3Runtime::loadLegacy(
          cfg.modelSoPath, cfg.weightPaths.front(), cfg.vocabPath, 512, 8194,
          1024, modelStatus.modelName);
    }
    modelStatus.state = ModelLoadState::Ready;
    modelStatus.modelName = runtime->modelName();
    modelStatus.contextLength = static_cast<int>(runtime->contextLength());
    modelStatus.message.clear();
  } catch (const std::exception &ex) {
    runtime.reset();
    modelStatus.state = ModelLoadState::Error;
    modelStatus.message = ex.what();
    throw;
  }
}

ModelStatus BgeM3EmbeddingModel::status() const {
  std::lock_guard<std::mutex> lock(mutex);
  return modelStatus;
}

EmbeddingResult BgeM3EmbeddingModel::embed(const EmbeddingRequest &request) {
  std::lock_guard<std::mutex> lock(mutex);
  if (!runtime || modelStatus.state != ModelLoadState::Ready)
    throw std::runtime_error("BgeM3EmbeddingModel: model is not loaded");
  if (request.input.empty())
    throw std::runtime_error("BgeM3EmbeddingModel: input text is empty");
  if (!request.model.empty() && request.model != modelStatus.modelName)
    throw std::runtime_error("BgeM3EmbeddingModel: requested model '" +
                             request.model + "' does not match loaded model '" +
                             modelStatus.modelName + "'");

  size_t tokenCount = 0;
  EmbeddingResult result;
  result.model = modelStatus.modelName;
  result.embedding = runtime->embed(request.input, &tokenCount);
  result.promptTokens = tokenCount;
  result.totalTokens = tokenCount;
  return result;
}

} // namespace runtime
} // namespace buddy
