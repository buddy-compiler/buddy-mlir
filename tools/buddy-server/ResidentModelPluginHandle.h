//===- ResidentModelPluginHandle.h - Resident plugin loader -----*- C++ -*-===//
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

#ifndef BUDDY_TOOLS_BUDDY_SERVER_RESIDENTMODELPLUGINHANDLE_H
#define BUDDY_TOOLS_BUDDY_SERVER_RESIDENTMODELPLUGINHANDLE_H

#include "buddy/runtime/core/ResidentModelPlugin.h"

#include <functional>
#include <memory>
#include <string>

namespace buddy {
namespace server {

class ResidentModelPluginHandle {
public:
  using ModelPtr =
      std::unique_ptr<buddy::runtime::ResidentModel,
                      std::function<void(buddy::runtime::ResidentModel *)>>;

  explicit ResidentModelPluginHandle(const std::string &pluginPath);
  ~ResidentModelPluginHandle();

  ResidentModelPluginHandle(const ResidentModelPluginHandle &) = delete;
  ResidentModelPluginHandle &
  operator=(const ResidentModelPluginHandle &) = delete;

  ModelPtr createModel() const;
  const std::string &modelType() const { return pluginModelType; }

private:
  std::string pluginPath;
  std::string pluginModelType;
  void *handle = nullptr;
  buddy::runtime::CreateResidentModelFn create = nullptr;
  buddy::runtime::DestroyResidentModelFn destroy = nullptr;
};

} // namespace server
} // namespace buddy

#endif // BUDDY_TOOLS_BUDDY_SERVER_RESIDENTMODELPLUGINHANDLE_H
