//===- MaskedLMModelPluginHandle.h - Masked-LM plugin loader -------------===//
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

#ifndef BUDDY_TOOLS_BUDDY_SERVER_MASKEDLMMODELPLUGINHANDLE_H
#define BUDDY_TOOLS_BUDDY_SERVER_MASKEDLMMODELPLUGINHANDLE_H

#include "buddy/runtime/core/MaskedLMModelPlugin.h"
#include <functional>
#include <memory>
#include <string>

namespace buddy {
namespace server {
class MaskedLMModelPluginHandle {
public:
  using ModelPtr =
      std::unique_ptr<buddy::runtime::MaskedLMModel,
                      std::function<void(buddy::runtime::MaskedLMModel *)>>;
  explicit MaskedLMModelPluginHandle(const std::string &path);
  ~MaskedLMModelPluginHandle();
  MaskedLMModelPluginHandle(const MaskedLMModelPluginHandle &) = delete;
  MaskedLMModelPluginHandle &
  operator=(const MaskedLMModelPluginHandle &) = delete;
  ModelPtr createModel() const;
  const std::string &modelType() const { return pluginModelType; }

private:
  std::string pluginPath, pluginModelType;
  void *handle = nullptr;
  std::shared_ptr<void> lifetime;
  buddy::runtime::CreateMaskedLMModelFn create = nullptr;
  buddy::runtime::DestroyMaskedLMModelFn destroy = nullptr;
};
} // namespace server
} // namespace buddy

#endif
