//===- ModelPluginDiscovery.h - Unified server plugin discovery -----------===//
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

#ifndef BUDDY_TOOLS_BUDDY_SERVER_MODELPLUGINDISCOVERY_H
#define BUDDY_TOOLS_BUDDY_SERVER_MODELPLUGINDISCOVERY_H

#include "buddy/runtime/core/ModelPlugin.h"

#include <string>

namespace buddy {
namespace server {

struct ModelPluginDiscoveryResult {
  std::string pluginPath;
  buddy::runtime::ModelServiceKind serviceKind =
      buddy::runtime::ModelServiceKind::Invalid;
};

ModelPluginDiscoveryResult discoverModelPlugin(const std::string &pluginPath);
const char *modelServiceKindName(buddy::runtime::ModelServiceKind kind);

} // namespace server
} // namespace buddy

#endif // BUDDY_TOOLS_BUDDY_SERVER_MODELPLUGINDISCOVERY_H
