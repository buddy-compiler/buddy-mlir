//===- ModelPluginDiscoveryFakePlugin.cpp - Discovery test plugin ---------===//
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

#include "buddy/runtime/core/ModelPlugin.h"

#if defined(BUDDY_DISCOVERY_REQUIRED_GLOBAL_SYMBOL)
#include <dlfcn.h>
#endif

#if !defined(BUDDY_DISCOVERY_MISSING_DESCRIPTOR)
extern "C" const buddy::runtime::ModelPluginDescriptorV1 *
buddy_model_plugin_descriptor_v1() {
#if defined(BUDDY_DISCOVERY_REQUIRED_GLOBAL_SYMBOL)
  if (!dlsym(RTLD_DEFAULT, "buddyRaxDependencyTestValue"))
    return nullptr;
#endif
  static const buddy::runtime::ModelPluginDescriptorV1 descriptor = {
      static_cast<uint32_t>(
#if defined(BUDDY_DISCOVERY_STRUCT_SIZE)
          BUDDY_DISCOVERY_STRUCT_SIZE
#else
          sizeof(buddy::runtime::ModelPluginDescriptorV1)
#endif
          ),
#if defined(BUDDY_DISCOVERY_ABI_VERSION)
      BUDDY_DISCOVERY_ABI_VERSION,
#else
      buddy::runtime::ModelPluginDescriptorAbiVersionV1,
#endif
#if defined(BUDDY_DISCOVERY_SERVICE_KIND)
      static_cast<buddy::runtime::ModelServiceKind>(
          BUDDY_DISCOVERY_SERVICE_KIND),
#else
      buddy::runtime::ModelServiceKind::Invalid,
#endif
#if defined(BUDDY_DISCOVERY_RESERVED)
      BUDDY_DISCOVERY_RESERVED};
#else
      0};
#endif
#if defined(BUDDY_DISCOVERY_NULL_DESCRIPTOR)
  (void)descriptor;
  return nullptr;
#else
  return &descriptor;
#endif
}
#endif
