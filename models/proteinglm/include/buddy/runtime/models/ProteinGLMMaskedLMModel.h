//===- ProteinGLMMaskedLMModel.h - ProteinGLM masked-LM model ------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_PROTEINGLM_MASKEDLMMODEL_H
#define BUDDY_RUNTIME_MODELS_PROTEINGLM_MASKEDLMMODEL_H

#include "buddy/runtime/core/MaskedLMModel.h"
#include <memory>

namespace buddy {
namespace runtime {

class ProteinGLMMaskedLMModel final : public MaskedLMModel {
public:
  ProteinGLMMaskedLMModel();
  ~ProteinGLMMaskedLMModel() override;
  void load(const MaskedLMModelConfig &config) override;
  ModelStatus status() const override;
  MaskedLMResult predict(const MaskedLMRequest &request) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace runtime
} // namespace buddy

#endif
